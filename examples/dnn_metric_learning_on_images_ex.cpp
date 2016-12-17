
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/misc_api.h>

using namespace dlib;
using namespace std;


std::vector<std::vector<string>> load_objects_list (
    const string& dir 
)
{
    std::vector<std::vector<string>> objects;
    for (auto subdir : directory(dir).get_dirs())
    {
        std::vector<string> imgs;
        for (auto img : subdir.get_files())
            imgs.push_back(img);

        objects.push_back(imgs);
    }
    return objects;
}

void load_mini_batch (
    const size_t num_ids,
    const size_t samples_per_id,
    dlib::rand& rnd,
    const std::vector<std::vector<string>>& objs,
    std::vector<matrix<rgb_pixel>>& images,
    std::vector<unsigned long>& labels
)
{
    images.clear();
    labels.clear();

    matrix<rgb_pixel> image; 
    for (size_t i = 0; i < num_ids; ++i)
    {
        const size_t id = rnd.get_random_32bit_number()%objs.size();
        for (size_t j = 0; j < samples_per_id; ++j)
        {
            const auto& obj = objs[id][rnd.get_random_32bit_number()%objs[id].size()];
            load_image(image, obj);
            images.push_back(std::move(image));
            labels.push_back(id);
        }
    }

    // You might want to do some data augmentation at this point.  Here we so some simple
    // color augmentation.
    for (auto&& crop : images)
        disturb_colors(crop,rnd);


    // All the images going into a mini-batch have to be the same size.  And really, all
    // the images in your entire training dataset should be the same size for what we are
    // doing to make the most sense.  
    DLIB_CASSERT(images.size() > 0);
    for (auto&& img : images)
    {
        DLIB_CASSERT(img.nr() == images[0].nr() && img.nc() == images[0].nc(), 
            "All the images in a single mini-batch must be the same size.");
    }
}


// ----------------------------------------------------------------------------------------

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;


template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;


// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level1 = res<512,res<512,res_down<512,SUBNET>>>;
template <typename SUBNET> using level2 = res<256,res<256,res<256,res<256,res<256,res_down<256,SUBNET>>>>>>;
template <typename SUBNET> using level3 = res<128,res<128,res<128,res_down<128,SUBNET>>>>;
template <typename SUBNET> using level4 = res<64,res<64,res<64,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,SUBNET>>>>>>;
template <typename SUBNET> using alevel3 = ares<128,ares<128,ares<128,ares_down<128,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<64,ares<64,ares<64,SUBNET>>>;

template <typename SUBNET> using final_pooling  = avg_pool_everything<SUBNET>;
template <typename SUBNET> using afinal_pooling  = avg_pool_everything<SUBNET>;

// training network type
using net_type = loss_metric<fc_no_bias<128,final_pooling<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<bn_con<con<64,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128,afinal_pooling<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<64,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Give folder as input.  It should contain sub-folders of images and we will " << endl;
        cout << "learn to distinguish these sub-folders with metric learning." << endl;
        return 1;
    }

    auto objs = load_objects_list(argv[1]);

    cout << "objs.size(): "<< objs.size() << endl;

    std::vector<matrix<rgb_pixel>> images;
    std::vector<unsigned long> labels;


    net_type net;

    dnn_trainer<net_type> trainer(net, sgd(0.0005, 0.9));
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("face_metric_sync", std::chrono::minutes(5));
    trainer.set_iterations_without_progress_threshold(300);

    // It's important to feed the GPU fast enough to keep it occupied.  So here we create a
    // bunch of threads that are responsible for creating mini-batches of training data.
    dlib::pipe<std::vector<matrix<rgb_pixel>>> qimages(4);
    dlib::pipe<std::vector<unsigned long>> qlabels(4);
    auto data_loader = [&qimages, &qlabels, &objs](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        std::vector<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;
        while(qimages.is_enabled())
        {
            try
            {
                load_mini_batch(15,15,rnd, objs, images, labels);
                qimages.enqueue(images);
                qlabels.enqueue(labels);
            }
            catch(std::exception& e)
            {
                cout << "EXCEPTION IN LOADING DATA" << endl;
                cout << e.what() << endl;
            }
        }
    };
    std::thread data_loader1([data_loader](){ data_loader(1); });
    std::thread data_loader2([data_loader](){ data_loader(2); });
    std::thread data_loader3([data_loader](){ data_loader(3); });
    std::thread data_loader4([data_loader](){ data_loader(4); });
    std::thread data_loader5([data_loader](){ data_loader(5); });


    // Here we do the training.  We keep passing mini-batches to the trainer until the
    // learning rate has dropped low enough.
    while(trainer.get_learning_rate() >= 1e-4)
    {
        qimages.dequeue(images);
        qlabels.dequeue(labels);
        trainer.train_one_step(images, labels);
    }

    // wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;

    // Save the network to disk
    net.clean();
    serialize("metric_network_renset.dat") << net;

    // stop all the data loading threads and wait for them to terminate.
    qimages.disable();
    qlabels.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();
    data_loader5.join();





    // Now, just to show an example of how you would use the network, lets check how well
    // it performs on the training data.
    dlib::rand rnd(time(0));
    load_mini_batch(15,15,rnd, objs, images, labels);

    // Run all the images through the network to get their vector embeddings.
    std::vector<matrix<float,0,1>> embedded = net(images);

    // Now, check if the embedding puts things with the same labels near each other and
    // things with different labels far apart.
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < embedded.size(); ++i)
    {
        for (size_t j = i+1; j < embedded.size(); ++j)
        {
            if (labels[i] == labels[j])
            {
                // The loss_metric layer will cause things with the same label to be less
                // than net.loss_details().get_distance_threshold() distance from each
                // other.  So we can use that distance value as our testing threshold.
                if (length(embedded[i]-embedded[j]) < net.loss_details().get_distance_threshold())
                    ++num_right;
                else
                    ++num_wrong;
            }
            else
            {
                if (length(embedded[i]-embedded[j]) >= net.loss_details().get_distance_threshold())
                    ++num_right;
                else
                    ++num_wrong;
            }
        }
    }

    cout << "num_right: "<< num_right << endl;
    cout << "num_wrong: "<< num_wrong << endl;

}


