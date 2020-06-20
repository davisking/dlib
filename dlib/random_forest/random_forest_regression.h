// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RANdOM_FOREST_REGRESSION_H_
#define DLIB_RANdOM_FOREST_REGRESSION_H_

#include "random_forest_regression_abstract.h"
#include <vector>
#include "../matrix.h"
#include <algorithm>
#include "../threads.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class dense_feature_extractor
    {

    public:
        typedef uint32_t feature;
        typedef matrix<double,0,1> sample_type;

        dense_feature_extractor(
        ) = default;

        void setup (
            const std::vector<sample_type>& x,
            const std::vector<double>& y 
        ) 
        {
            DLIB_CASSERT(x.size() > 0);
            DLIB_CASSERT(x.size() == y.size());
            for (auto& el : x)
                DLIB_CASSERT(el.size() == x[0].size(), "All the vectors in a training set have to have the same dimensionality.");

            DLIB_CASSERT(x[0].size() != 0, "The vectors can't be empty.");

            num_feats = x[0].size();
        }


        void get_random_features (
            dlib::rand& rnd,
            size_t num,
            std::vector<feature>& feats
        ) const
        {
            DLIB_ASSERT(max_num_feats() != 0);
            num = std::min(num, num_feats);

            feats.clear();
            for (size_t i = 0; i < num_feats; ++i)
                feats.push_back(i);

            // now pick num features at random
            for (size_t i = 0; i < num; ++i)
            {
                auto idx = rnd.get_integer_in_range(i,num_feats);
                std::swap(feats[i], feats[idx]);
            }
            feats.resize(num);
        }

        double extract_feature_value (
            const sample_type& item,
            const feature& f
        ) const
        {
            DLIB_ASSERT(max_num_feats() != 0);
            return item(f);
        }

        size_t max_num_feats (
        ) const
        {
            return num_feats;
        }

        friend void serialize(const dense_feature_extractor& item, std::ostream& out)
        {
            serialize("dense_feature_extractor", out);
            serialize(item.num_feats, out);
        }

        friend void deserialize(dense_feature_extractor& item, std::istream& in)
        {
            check_serialized_version("dense_feature_extractor", in);
            deserialize(item.num_feats, in);
        }

    private:
        size_t num_feats = 0;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    struct internal_tree_node
    {
        uint32_t left;
        uint32_t right;
        float split_threshold;
        typename feature_extractor::feature split_feature;
    };

    template <typename feature_extractor>
    void serialize(const internal_tree_node<feature_extractor>& item, std::ostream& out)
    {
        serialize(item.left, out);
        serialize(item.right, out);
        serialize(item.split_threshold, out);
        serialize(item.split_feature, out);
    }

    template <typename feature_extractor>
    void deserialize(internal_tree_node<feature_extractor>& item, std::istream& in)
    {
        deserialize(item.left, in);
        deserialize(item.right, in);
        deserialize(item.split_threshold, in);
        deserialize(item.split_feature, in);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor = dense_feature_extractor
        >
    class random_forest_regression_function
    {

    public:

        typedef feature_extractor feature_extractor_type;
        typedef typename feature_extractor::sample_type sample_type;

        random_forest_regression_function(
        ) = default;

        random_forest_regression_function (
            feature_extractor_type&& fe_,
            std::vector<std::vector<internal_tree_node<feature_extractor>>>&& trees_,
            std::vector<std::vector<float>>&& leaves_
        ) :
            fe(std::move(fe_)),
            trees(std::move(trees_)),
            leaves(std::move(leaves_))
        {
            DLIB_ASSERT(trees.size() > 0);
            DLIB_ASSERT(trees.size() == leaves.size(), "Every set of tree nodes has to have leaves");
#ifdef ENABLE_ASSERTS
            for (size_t i = 0; i < trees.size(); ++i)
            {
                DLIB_ASSERT(trees[i].size() > 0, "A tree can't have 0 leaves.");
                for (auto& node : trees[i])
                {
                    DLIB_ASSERT(trees[i].size()+leaves[i].size() > node.left, "left node index in tree is too big. There is no associated tree node or leaf.");
                    DLIB_ASSERT(trees[i].size()+leaves[i].size() > node.right, "right node index in tree is too big. There is no associated tree node or leaf.");
                }
            }
#endif
        }

        size_t get_num_trees(
        ) const
        {
            return trees.size();
        }

        const std::vector<std::vector<internal_tree_node<feature_extractor>>>& get_internal_tree_nodes (
        ) const { return trees; }

        const std::vector<std::vector<float>>& get_tree_leaves (
        ) const { return leaves; }

        const feature_extractor_type& get_feature_extractor (
        ) const { return fe; }

        double operator() (
            const sample_type& x
        ) const
        {
            DLIB_ASSERT(get_num_trees() > 0);

            double accum = 0;

            for (size_t i = 0; i < trees.size(); ++i)
            {
                auto& tree = trees[i];
                // walk the tree to the leaf
                uint32_t idx = 0;
                while(idx < tree.size())
                {
                    auto feature_value = fe.extract_feature_value(x, tree[idx].split_feature);
                    if (feature_value < tree[idx].split_threshold)
                        idx = tree[idx].left;
                    else
                        idx = tree[idx].right;
                }
                // compute leaf index 
                accum += leaves[i][idx-tree.size()];
            }

            return accum/trees.size();
        }

        friend void serialize(const random_forest_regression_function& item, std::ostream& out)
        {
            serialize("random_forest_regression_function", out);
            serialize(item.fe, out);
            serialize(item.trees, out);
            serialize(item.leaves, out);
        }

        friend void deserialize(random_forest_regression_function& item, std::istream& in)
        {
            check_serialized_version("random_forest_regression_function", in);
            deserialize(item.fe, in);
            deserialize(item.trees, in);
            deserialize(item.leaves, in);
        }

    private:

        /*!
            CONVENTION
                - trees.size() == leaves.size()
                - Any .left or .right index in trees that is larger than the number of
                  nodes in the tree references a leaf. Moreover, the index of the leaf is
                  computed by subtracting the number of nodes in the tree.
        !*/

        feature_extractor_type fe;

        // internal nodes of trees
        std::vector<std::vector<internal_tree_node<feature_extractor>>> trees;
        // leaves of trees
        std::vector<std::vector<float>> leaves;

    };

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor = dense_feature_extractor
        >
    class random_forest_regression_trainer
    {
    public:
        typedef feature_extractor feature_extractor_type;
        typedef random_forest_regression_function<feature_extractor> trained_function_type;
        typedef typename feature_extractor::sample_type sample_type;


        random_forest_regression_trainer (
        ) = default;

        const feature_extractor_type& get_feature_extractor (
        ) const
        {
            return fe_;
        }

        void set_feature_extractor (
            const feature_extractor_type& feat_extractor
        )
        {
            fe_ = feat_extractor;
        }

        void set_seed (
            const std::string& seed
        )
        {
            random_seed = seed;
        }

        const std::string& get_random_seed (
        ) const
        {
            return random_seed;
        }

        size_t get_num_trees (
        ) const
        {
            return num_trees;
        }

        void set_num_trees (
            size_t num
        )
        {
            DLIB_CASSERT(num > 0);
            num_trees = num;
        }

        void set_feature_subsampling_fraction (
            double frac
        )
        {
            DLIB_CASSERT(0 < frac && frac <= 1);
            feature_subsampling_frac = frac;
        }

        double get_feature_subsampling_frac(
        ) const
        {
            return feature_subsampling_frac;
        }

        void set_min_samples_per_leaf (
            size_t num
        )
        {
            DLIB_ASSERT(num > 0);
            min_samples_per_leaf = num;
        }

        size_t get_min_samples_per_leaf(
        ) const
        {
            return min_samples_per_leaf;
        }

        void be_verbose (
        )
        {
            verbose = true;
        }

        void be_quiet (
        )
        {
            verbose = false;
        }

        trained_function_type train (
            const std::vector<sample_type>& x,
            const std::vector<double>& y 
        ) const
        {
            std::vector<double> junk; 
            return do_train(x,y,junk,false);
        }

        trained_function_type train (
            const std::vector<sample_type>& x,
            const std::vector<double>& y,
            std::vector<double>& oob_values 
        ) const
        {
            return do_train(x,y,oob_values,true);
        }

    private:

        trained_function_type do_train (
            const std::vector<sample_type>& x,
            const std::vector<double>& y,
            std::vector<double>& oob_values, 
            bool compute_oob_values
        ) const 
        {
            DLIB_CASSERT(x.size() == y.size());
            DLIB_CASSERT(x.size() > 0);

            feature_extractor_type fe = fe_;
            fe.setup(x,y);

            DLIB_CASSERT(fe.max_num_feats() != 0);

            std::vector<std::vector<internal_tree_node<feature_extractor>>> all_trees(num_trees);
            std::vector<std::vector<float>> all_leaves(num_trees);

            const size_t feats_per_node = std::max(1.0,std::round(fe.max_num_feats()*feature_subsampling_frac));

            // Each tree couldn't have more than this many interior nodes.  It might
            // end up having less though. We need to know this value because the way
            // we mark a left or right pointer in a tree as pointing to a leaf is by
            // making its index larger than the number of interior nodes in the tree.
            // But we don't know the tree's size before we finish building it.  So we
            // will use max_num_nodes as a proxy during tree construction and then go
            // back and fix it once a tree's size is known.
            const uint32_t max_num_nodes = y.size(); 

            std::vector<uint32_t> oob_hits;

            if (compute_oob_values)
            {
                oob_values.resize(y.size());
                oob_hits.resize(y.size());
            }


            std::mutex m;

            // Calling build_tree(i) creates the ith tree and stores the results in
            // all_trees and all_leaves.
            auto build_tree = [&](long i)
            {
                dlib::rand rnd(random_seed + std::to_string(i));
                auto& tree = all_trees[i];
                auto& leaves = all_leaves[i];

                // Check if there are fewer than min_samples_per_leaf and if so then
                // don't make any tree.  Just average the things and be done. 
                if (y.size() <= min_samples_per_leaf)
                {
                    leaves.push_back(mean(mat(y)));
                    return;
                }


                double sumy = 0;
                // pick a random bootstrap of the data.
                std::vector<std::pair<float,uint32_t>> idxs(y.size());
                for (auto& idx : idxs) {
                    idx = std::make_pair(0.0f, static_cast<uint32_t>(rnd.get_integer(y.size())));
                    sumy += y[idx.second];
                }

                // We are going to use ranges_to_process as a stack that tracks which
                // range of samples we are going to split next.
                std::vector<range_t> ranges_to_process;
                // start with the root of the tree, i.e. the entire range of training
                // samples.
                ranges_to_process.emplace_back(sumy, 0, static_cast<uint32_t>(y.size()));
                // push an unpopulated root node into the tree.  We will populate it
                // when we process its corresponding range. 
                tree.emplace_back();

                std::vector<typename feature_extractor::feature> feats;

                while(ranges_to_process.size() > 0)
                {
                    // Grab the next range/node to process.
                    const auto range = ranges_to_process.back();
                    ranges_to_process.pop_back();


                    // Get the split features we will consider at this node.
                    fe.get_random_features(rnd, feats_per_node, feats);
                    // Then find the best split
                    auto best_split = find_best_split_among_feats(fe, range, feats, x, y, idxs); 

                    range_t left_split(best_split.left_sum, range.begin, best_split.split_idx);
                    range_t right_split(best_split.right_sum, best_split.split_idx, range.end);

                    DLIB_ASSERT(left_split.begin < left_split.end);
                    DLIB_ASSERT(right_split.begin < right_split.end);

                    // Now that we know the split we can populate the parent node we popped
                    // from ranges_to_process.
                    tree[range.tree_idx].split_threshold = best_split.split_threshold; 
                    tree[range.tree_idx].split_feature = best_split.split_feature; 

                    // If the left split is big enough to make a new interior leaf
                    // node. We also stop splitting if all the samples went into this node.
                    // This could happen if the features are all uniform so there just
                    // isn't any way to split them anymore.
                    if (left_split.size() > min_samples_per_leaf && right_split.size() != 0)
                    {
                        // allocate an interior leaf node for it.
                        left_split.tree_idx = tree.size();
                        tree.emplace_back(); 
                        // set the pointer in the parent node to the newly allocated
                        // node.
                        tree[range.tree_idx].left  = left_split.tree_idx;

                        ranges_to_process.emplace_back(left_split);
                    }
                    else
                    {
                        // Add to leaves.  Don't forget to set the pointer in the
                        // parent node to the newly allocated leaf node.
                        tree[range.tree_idx].left = leaves.size() + max_num_nodes;
                        leaves.emplace_back(static_cast<float>(left_split.avg()));
                    }


                    // If the right split is big enough to make a new interior leaf
                    // node. We also stop splitting if all the samples went into this node.
                    // This could happen if the features are all uniform so there just
                    // isn't any way to split them anymore.
                    if (right_split.size() > min_samples_per_leaf && left_split.size() != 0)
                    {
                        // allocate an interior leaf node for it.
                        right_split.tree_idx = tree.size();
                        tree.emplace_back(); 
                        // set the pointer in the parent node to the newly allocated
                        // node.
                        tree[range.tree_idx].right  = right_split.tree_idx;

                        ranges_to_process.emplace_back(right_split);
                    }
                    else
                    {
                        // Add to leaves.  Don't forget to set the pointer in the
                        // parent node to the newly allocated leaf node.
                        tree[range.tree_idx].right = leaves.size() + max_num_nodes;
                        leaves.emplace_back(static_cast<float>(right_split.avg()));
                    }
                } // end while (still building tree)

                // Fix the leaf pointers in the tree now that we know the correct
                // tree.size() value.
                DLIB_CASSERT(max_num_nodes >= tree.size()); 
                const auto offset = max_num_nodes - tree.size();
                for (auto& n : tree)
                {
                    if (n.left >= max_num_nodes)
                        n.left -= offset;
                    if (n.right >= max_num_nodes)
                        n.right -= offset;
                }
                

                if (compute_oob_values)
                {
                    std::sort(idxs.begin(), idxs.end(), 
                        [](const std::pair<float,uint32_t>& a, const std::pair<float,uint32_t>& b) {return a.second<b.second; });

                    std::lock_guard<std::mutex> lock(m);

                    size_t j = 0;
                    for (size_t i = 0; i < oob_values.size(); ++i)
                    {
                        // check if i is in idxs
                        while(j < idxs.size() && i > idxs[j].second)
                            ++j;

                        // i isn't in idxs so it's an oob sample and we should process it.
                        if (j == idxs.size() || idxs[j].second != i)
                        {
                            oob_hits[i]++;

                            // walk the tree to find the leaf value for this oob sample
                            uint32_t idx = 0;
                            while(idx < tree.size())
                            {
                                auto feature_value = fe.extract_feature_value(x[i], tree[idx].split_feature);
                                if (feature_value < tree[idx].split_threshold)
                                    idx = tree[idx].left;
                                else
                                    idx = tree[idx].right;
                            }
                            oob_values[i] += leaves[idx-tree.size()];
                        }
                    }
                }
            };

            if (verbose)
                parallel_for_verbose(0, num_trees, build_tree);
            else
                parallel_for(0, num_trees, build_tree);


            if (compute_oob_values)
            {
                double meanval = 0;
                double cnt = 0;
                for (size_t i = 0; i < oob_values.size(); ++i)
                {
                    if (oob_hits[i] != 0)
                    {
                        oob_values[i] /= oob_hits[i];
                        meanval += oob_values[i];
                        ++cnt;
                    }
                }

                // If there are some elements that didn't get hits, we set their oob values
                // to the mean oob value.
                if (cnt != 0)
                {
                    const double typical_value = meanval/cnt;
                    for (size_t i = 0; i < oob_values.size(); ++i)
                    {
                        if (oob_hits[i] == 0)
                            oob_values[i] = typical_value;
                    }
                }
            }

            return trained_function_type(std::move(fe), std::move(all_trees), std::move(all_leaves));
        }

        struct range_t 
        {
            range_t(
                double sumy,
                uint32_t begin,
                uint32_t end
            ) : sumy(sumy), begin(begin), end(end), tree_idx(0) {}

            double sumy;
            uint32_t begin;
            uint32_t end;

            // Every range object corresponds to an entry in a tree. This tells you the
            // tree node that owns the range.
            uint32_t tree_idx; 

            uint32_t size() const { return end-begin; }
            double avg() const { return sumy/size(); }
        };

        struct best_split_details
        {
            double score = -std::numeric_limits<double>::infinity();
            double left_sum;
            double right_sum;
            uint32_t split_idx;
            double split_threshold;
            typename feature_extractor::feature split_feature;

            bool operator < (const best_split_details& rhs) const
            {
                return score < rhs.score;
            }
        };

        static best_split_details find_best_split (
            const range_t& range,
            const std::vector<double>& y,
            const std::vector<std::pair<float,uint32_t>>& idxs
        )
        /*!
            requires
                - max(mat(idxs)) < y.size()
                - range.sumy == sum of y[idxs[j].second] for all valid j in range [range.begin, range.end). 
            ensures
                - finds a threshold T such that there exists an i satisfying the following:
                    - y[idxs[j].second] < T for all j <= i
                    - y[idxs[j].second] > T for all j > i
                  Therefore, the threshold T partitions the contents of y into two groups,
                  relative to the ordering established by idxs.  Moreover the partitioning
                  of y values into two groups has the additional requirement that it is
                  optimal in the sense that the sum of the squared deviations from each
                  partition's mean is minimized.
        !*/
        {

            size_t best_i = range.begin;
            double best_score = -1;
            double left_sum = 0;
            double best_left_sum = y[idxs[range.begin].second];
            const auto size = range.size();
            size_t left_size = 0;
            for (size_t i = range.begin; i+1 < range.end; ++i)
            {
                ++left_size;
                left_sum += y[idxs[i].second];

                // Don't split here because the next element has the same feature value so
                // we can't *really* split here.
                if (idxs[i].first==idxs[i+1].first)
                    continue;

                const double right_sum = range.sumy-left_sum;

                const double score = left_sum*left_sum/left_size + right_sum*right_sum/(size-left_size);

                if (score > best_score)
                {
                    best_score = score;
                    best_i = i;
                    best_left_sum = left_sum;
                }
            }

            best_split_details result;
            result.score = best_score;
            result.left_sum = best_left_sum;
            result.right_sum = range.sumy-best_left_sum;
            result.split_idx = best_i+1; // one past the end of the left range
            result.split_threshold = (idxs[best_i].first+idxs[best_i+1].first)/2;

            return result;
        }


        static best_split_details find_best_split_among_feats(
            const feature_extractor& fe,
            const range_t& range, 
            const std::vector<typename feature_extractor::feature>& feats, 
            const std::vector<sample_type>& x,
            const std::vector<double>& y,
            std::vector<std::pair<float,uint32_t>>& idxs
        )
        {
            auto compare_first = [](const std::pair<float,uint32_t>& a, const std::pair<float,uint32_t>& b) { return a.first<b.first; };
            best_split_details best;
            for (auto& feat : feats)
            {
                // Extract feature values for this feature and sort the indexes based on
                // that feature so we can then find the best split.
                for (auto i = range.begin; i < range.end; ++i)
                    idxs[i].first = fe.extract_feature_value(x[idxs[i].second], feat);

                std::stable_sort(idxs.begin()+range.begin, idxs.begin()+range.end, compare_first);

                auto split = find_best_split(range, y, idxs);

                if (best < split)
                {
                    best = split;
                    best.split_feature = feat;
                }
            }

            // resort idxs based on winning feat
            for (auto i = range.begin; i < range.end; ++i)
                idxs[i].first = fe.extract_feature_value(x[idxs[i].second], best.split_feature);
            std::stable_sort(idxs.begin()+range.begin, idxs.begin()+range.end, compare_first);

            return best;
        }

        std::string random_seed;
        size_t num_trees = 1000;
        double feature_subsampling_frac = 1.0/3.0;
        size_t min_samples_per_leaf = 5;
        feature_extractor_type fe_;
        bool verbose = false;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RANdOM_FOREST_REGRESSION_H_


