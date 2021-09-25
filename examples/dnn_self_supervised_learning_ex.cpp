#include <dlib/dnn.h>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;

ostream& operator<<(ostream& out, const tensor& t)
{
    out << t.num_samples() << 'x' << t.k() << 'x' << t.nr() << 'x' << t.nc();
    return out;
}

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class input_rgb_image_pair
    {
    public:
        typedef std::pair<matrix<rgb_pixel>, matrix<rgb_pixel>> input_type;

        input_rgb_image_pair (
        ) :
            avg_red_first(122.782),
            avg_green_first(117.001),
            avg_blue_first(104.298),
            avg_red_second(122.782),
            avg_green_second(117.001),
            avg_blue_second(104.298)
        {
        }

        input_rgb_image_pair (
            float avg_red_first,
            float avg_green_first,
            float avg_blue_first
        ) :
            avg_red_first(avg_red_first),
            avg_green_first(avg_green_first),
            avg_blue_first(avg_blue_first),
            avg_red_second(avg_red_first),
            avg_green_second(avg_green_first),
            avg_blue_second(avg_blue_first)
        {}

        input_rgb_image_pair (
            float avg_red_first,
            float avg_green_first,
            float avg_blue_first,
            float avg_red_second,
            float avg_green_second,
            float avg_blue_second
        ) :
            avg_red_first(avg_red_first),
            avg_green_first(avg_green_first),
            avg_blue_first(avg_blue_first),
            avg_red_second(avg_red_second),
            avg_green_second(avg_green_second),
            avg_blue_second(avg_blue_second)
        {}

        inline input_rgb_image_pair (
            const input_rgb_image& item
        ) :
            avg_red_first(item.get_avg_red()),
            avg_green_first(item.get_avg_green()),
            avg_blue_first(item.get_avg_blue()),
            avg_red_second(item.get_avg_red()),
            avg_green_second(item.get_avg_green()),
            avg_blue_second(item.get_avg_blue())
        {}

        template <size_t NR, size_t NC>
        inline input_rgb_image_pair (
            const input_rgb_image_sized<NR, NC>& item
        ) :
            avg_red_first(item.get_avg_red()),
            avg_green_first(item.get_avg_green()),
            avg_blue_first(item.get_avg_blue()),
            avg_red_second(item.get_avg_red()),
            avg_green_second(item.get_avg_green()),
            avg_blue_second(item.get_avg_blue())
        {}

        float get_avg_red_first()   const { return avg_red_first; }
        float get_avg_green_first() const { return avg_green_first; }
        float get_avg_blue_first()  const { return avg_blue_first; }
        float get_avg_red_second()   const { return avg_red_second; }
        float get_avg_green_second() const { return avg_green_second; }
        float get_avg_blue_second()  const { return avg_blue_second; }

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin, iend) > 0);
            const auto nr = ibegin->first.nr();
            const auto nc = ibegin->first.nc();

            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->first.nr() == nr && i->first.nc()==nc &&
                             i->second.nr() == nr && i->second.nc() == nc,
                    "\t input_rgb_image_pair::to_tensor()"
                    << "\n\t All matrices given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->first.nr(): " << i->first.nr()
                    << "\n\t i->first.nc(): " << i->first.nc()
                    << "\n\t i->second.nr(): " << i->second.nr()
                    << "\n\t i->second.nc(): " << i->second.nc()
                );
            }

            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(2 * std::distance(ibegin, iend), 3, nr, nc);

            const size_t offset = nr * nc;
            const size_t offset2 = data.size() / 2;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        rgb_pixel temp_first = i->first(r, c);
                        rgb_pixel temp_second = i->second(r, c);
                        auto p = ptr++;
                        *p = (temp_first.red - avg_red_first) / 256.0;
                        *(p + offset2) = (temp_second.red - avg_red_second) / 256.0;
                        p += offset;
                        *p = (temp_first.green - avg_green_first) / 256.0;
                        *(p + offset2) = (temp_second.green - avg_green_second) / 256.0;
                        p += offset;
                        *p = (temp_first.blue - avg_blue_first) / 256.0;
                        *(p + offset2) = (temp_second.blue - avg_blue_second) / 256.0;
                        p += offset;
                    }
                }
                ptr += offset * (data.k() - 1);
            }
        }

        friend void serialize(const input_rgb_image_pair& item, std::ostream& out)
        {
            serialize("input_rgb_image_pair", out);
            serialize(item.avg_red_first, out);
            serialize(item.avg_green_first, out);
            serialize(item.avg_blue_first, out);
            serialize(item.avg_red_second, out);
            serialize(item.avg_green_second, out);
            serialize(item.avg_blue_second, out);
        }

        friend void deserialize(input_rgb_image_pair& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_rgb_image_pair" && version != "input_rgb_image" && version != "input_rgb_image_sized")
                throw serialization_error("Unexpected version found while deserializing dlib::input_rgb_image_pair.");

            deserialize(item.avg_red_first, in);
            deserialize(item.avg_green_first, in);
            deserialize(item.avg_blue_first, in);
            if (version == "input_rgb_image_pair")
            {
                deserialize(item.avg_red_second, in);
                deserialize(item.avg_green_second, in);
                deserialize(item.avg_blue_second, in);
            }
            else
            {
                item.avg_red_second = item.avg_red_first;
                item.avg_green_second = item.avg_green_first;
                item.avg_blue_second = item.avg_blue_first;
            }
            // read and discard the sizes if this was really a sized input layer.
            if (version == "input_rgb_image_sized")
            {
                size_t nr, nc;
                deserialize(nr, in);
                deserialize(nc, in);
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const input_rgb_image_pair& item)
        {
            out << "input_rgb_image_pair(("
                << item.avg_red_first<<","<<item.avg_green_first<<","<<item.avg_blue_first << "),("
                << item.avg_red_second<<","<<item.avg_green_second<<","<<item.avg_blue_second << "))";
            return out;
        }

        friend void to_xml(const input_rgb_image_pair& item, std::ostream& out)
        {
            out << "<input_rgb_image_pair r1='"<<item.avg_red_first<<"' g1='"<<item.avg_green_first<<"' b1='"<<item.avg_blue_first
                << ",r2='"<<item.avg_red_second<<"' g2='"<<item.avg_green_second<<"' b2='"<<item.avg_blue_second<<"'/>";
        }

    private:
        float avg_red_first, avg_red_second;
        float avg_green_first, avg_green_second;
        float avg_blue_first, avg_blue_second;
    };
}

namespace dlib
{
    class loss_barlow_twins_
    {
    public:

        loss_barlow_twins_() = default;

        loss_barlow_twins_(float lambda) : lambda(lambda)
        {
            DLIB_CASSERT(lambda > 0);
        }

        template <
            typename SUBNET
        >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 2);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples() % sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.nr() == 1 && output_tensor.nc() == 1);
            DLIB_CASSERT(grad.nr() == 1 && grad.nc() == 1);

            const auto batch_size = output_tensor.num_samples() / 2;
            const auto sample_size = output_tensor.k();
            // std::cout << "batch size: " << batch_size << std::endl;
            // std::cout << "sample size: " << sample_size << std::endl;

            // some alias helpers to access the samples in the batch
            alias_tensor split(batch_size, sample_size);
            alias_tensor sample(1, sample_size);
            auto za = split(output_tensor);
            auto zb = split(output_tensor, batch_size);

            // normalize both batches independently across the batch dimension
            const double eps = DEFAULT_BATCH_NORM_EPS;
            resizable_tensor temp, za_norm, zb_norm, means, invstds, running_means, running_variances, gamma, beta;
            gamma.set_size(1, sample_size);
            beta.set_size(1, sample_size);
            gamma = 1;
            beta = 0;
            // std::cout << "batch_normalize a" << std::endl;
            tt::batch_normalize(eps, za_norm, means, invstds, 1, running_means, running_variances, za, gamma, beta);
            // std::cout << "batch_normalize b" << std::endl;
            tt::batch_normalize(eps, zb_norm, means, invstds, 1, running_means, running_variances, zb, gamma, beta);

            // compute the empirical cross-correlation matrix
            temp.set_size(sample_size, sample_size);
            tt::gemm(0, temp, 1, za_norm, true, zb_norm, false);
            temp /= batch_size;

            const matrix<float> A = mat(za_norm);
            const matrix<float> B = mat(zb_norm);
            const matrix<float> C = mat(temp);  // trans(A) * B
            const matrix<float> D = ones_matrix<float>(sample_size, sample_size) - identity_matrix<float>(sample_size);
            // std::cout << diag(C) << std::endl;
            // std::cout << D << std::endl;
            // std::cout << C - D << std::endl;

            // compute the loss: notation from http://www.matrixcalculus.org/
            // diagonal: sum((diag(A' * B) - vector(1)).^2)
            // ----------------------------------------
            // 	=> d/dA = 2 * B * diag(diag(A' * B) - vector(1))
            // 	=> d/dB = 2 * A * diag(diag(A' * B) - vector(1))
            const matrix<float> GDA = 2 * (B * diagm(diag(C) - 1));
            const matrix<float> GDB = 2 * (A * diagm(diag(C) - 1));

            // off-diag: sum(((A'* B) .* D).^2)
            // -----------------------------
            //  => d/dA = 2 * B * ((B' * A) .* (D .* D)') = 2 * B * (C' .* (D .* D)')
            //  => d/dB = 2 * A * ((A' * B) .* (D .* D)) = 2 * A * (C .* (D .* D))
            const matrix<float> GOA = 2 * B * pointwise_multiply(trans(C), trans(squared(D)));
            const matrix<float> GOB = 2 * A * pointwise_multiply(C, squared(D));
            // std::cout << GDA.nr() << 'x' << GDB.nc() << std::endl;
            // std::cout << GDB.nr() << 'x' << GDB.nc() << std::endl;
            // std::cout << GOA.nr() << 'x' << GDB.nc() << std::endl;
            // std::cout << GOB.nr() << 'x' << GDB.nc() << std::endl;

            auto g = grad.host();
            const auto offset = batch_size * sample_size;
            for (long r = 0; r < batch_size; ++r)
            {
                for (long c = 0; c < sample_size; ++c)
                {
                    const size_t idx = tensor_index(za_norm, r, c, 0, 0);
                    g[idx] = GDA(r, c) + lambda * GOA(r, c);
                    g[idx + offset] = GDB(r, c) + lambda * GOB(r, c);
                }
            }

            double diagonal_loss = sum(squared(diag(C) - 1));
            double off_diag_loss = sum(squared(C - diagm(diag(C))));

            return diagonal_loss + lambda * off_diag_loss;
        }

        friend void serialize(const loss_barlow_twins_& item, std::ostream& out)
        {
            serialize("loss_barlow_twins_", out);
            serialize(item.lambda, out);
        }

        friend void deserialize(loss_barlow_twins_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "loss_barlow_twins_")
            {
                deserialize(item.lambda, in);
            }
            else
            {
                throw serialization_error("Unexpected version found while deserializing dlib::loss_barlow_twins_.  Instead found " + version);
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_barlow_twins_& item)
        {
            out << "loss_barlow_twins (lambda=" << item.lambda << ")";
            return out;
        }

        friend void to_xml(const loss_barlow_twins_& item, std::ostream& out)
        {
            out << "<loss_barlow_twins lambda='" << item.lambda << "'/>";
        }

    private:
        float lambda = 0.0051;
    };

    template <typename SUBNET>
    using loss_barlow_twins = add_loss_layer<loss_barlow_twins_, SUBNET>;
}

int main (const int argc, const char** argv)
try
{
    std::vector<std::pair<matrix<rgb_pixel>, matrix<rgb_pixel>>> batch;

    using net_type = loss_barlow_twins<
                     fc_no_bias<512, relu<bn_fc<fc_no_bias<512, relu<bn_fc<fc_no_bias<512,
                     avg_pool_everything<
                     relu<bn_con<con<128, 3, 3, 2, 2,
                     relu<bn_con<con<64, 3, 3, 2, 2,
                     relu<bn_con<con<32, 3, 3, 2, 2,
                     input_rgb_image_pair>>>>>>>>>>>>>>>>>>;
    std::vector<matrix<rgb_pixel>> training_images, testing_images;
    std::vector<unsigned long> training_labels, testing_labels;
    load_cifar_10_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);
    cout << "loaded CIFAR-10 training images: " << training_images.size() << endl;
    net_type net(loss_barlow_twins_(0.0051));
    disable_duplicative_biases(net);

    // dnn_trainer<net_type> trainer(net);
    // trainer.be_verbose();
    dlib::rand rnd;
    std::vector<sgd> solvers(net.num_computational_layers, sgd());
    while (true)
    {
        batch.clear();
        while (batch.size() < 64)
        {
            const auto idx = rnd.get_random_64bit_number() % training_images.size();
            auto image = training_images[idx];
            batch.emplace_back(image, fliplr(image));
        }
        resizable_tensor x;
        // cout << trainer.get_train_one_step_calls() << ": " << trainer.get_average_loss() << endl;
        // trainer.train_one_step(batch);
        net.to_tensor(batch.begin(), batch.end(), x);
        double loss = net.compute_loss(x);
        cout << "loss: " << loss << endl;
        // cout << "back propagating error" << endl;
        net.back_propagate_error(x);
        // cout << "updating network parameters" << endl;
        net.update_parameters(solvers, 1e-5);
        // cout << net << endl;
        cin.get();
    }
    return EXIT_SUCCESS;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
