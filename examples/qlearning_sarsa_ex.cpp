#include <dlib/control/qlearning.h>
#include <dlib/matrix.h>
#include <limits>
#include <cmath>
#include <vector>
#include <iostream>

using namespace dlib;

template <
        typename state_type,
        typename action_type
        >
class feature_extractor
{
public:
    feature_extractor(
        unsigned int h,
        unsigned int w,
        unsigned int na
    ) : height(h), width(w), num_actions(na) {}


    inline long num_features(
    ) const { return num_actions * height * width; }

    matrix<double,0,1> get_features(
        const state_type &state,
        const action_type &action
    ) const
    {
        matrix<double,0,1> feats(num_features());
        feats = 0;
        //for(auto i = 0u; i < num_actions; i++)
        //    feats(num_actions * state + i) = 1;
        feats(num_actions*state + static_cast<int>(action)) = 1;

        return feats;
    }

private:
    int height, width, num_actions;
};

template <
        int height,
        int width,
        template<typename,typename> class feature_extractor_type
        >
class cliff_model
{
public:
    enum class actions {up = 0, right, down, left};
    constexpr static double EPS = 1e-16;

    typedef int state_type;
    typedef actions action_type;
    typedef int reward_type;

    typedef feature_extractor_type<state_type, action_type> feature_extractor;

    explicit cliff_model(
    ) : fe(height, width, 4){}

    action_type random_action(
        const state_type& state
    ) const
    {
        std::uniform_int_distribution<int> dist(0,3);
        return static_cast<action_type>(dist(gen));
    }

    action_type find_best_action(
        const state_type& state,
        const matrix<double,0,1>& w
    ) const
    {
        auto best = std::numeric_limits<double>::lowest();
        auto best_indexes = std::vector<int>();

        for(auto i = 0; i < 4; i++){
            auto feats = get_features(state, static_cast<action_type>(i));
            auto product = dot(w, feats);
            if(product > best){
                best = product;
                best_indexes.clear();
            }

            if(std::abs(product - best) < EPS)
                best_indexes.push_back(i);
        }

        std::uniform_int_distribution<unsigned long> dist(0, best_indexes.size()-1);
        return static_cast<action_type>(best_indexes[dist(gen)]);
    }

    const feature_extractor& get_feature_extractor(
    ) const { return fe; }

    auto states_size(
    ) const -> decltype(get_feature_extractor().num_features())
    {
        return get_feature_extractor().num_features();
    }

    auto get_features(
        const state_type &state,
        const action_type &action
    ) const -> decltype(get_feature_extractor().get_features(state, action))
    { return get_feature_extractor().get_features(state, action); }

    reward_type reward(
        const state_type &state,
        const action_type &action,
        const state_type &new_state
    ) const
    {
        return !is_final(new_state) ? -1 : is_success(new_state) ? 100 : -100;
    }

    state_type initial_state(
    ) const { return static_cast<state_type>((height-1) * width); }

    state_type step(
        const state_type& state,
        const action_type& action
    ) const
    {
        if(out_of_bounds(state, action))
            return state;

        return action == actions::up    ?   state - width   :
               action == actions::down  ?   state + width   :
               action == actions::right ?   state + 1       :
                                            state - 1       ;
    }

    bool is_success(
        const state_type &state
    ) const { return state == height*width - 1; }

    bool is_failure(
        const state_type &state
    ) const { return state/width == height-1 && state%width > 0 && state%width < width-1;}

    bool is_final(
        const state_type& state
    ) const { return is_success(state) || is_failure(state); }

private:
    bool out_of_bounds(
        const state_type& state,
        const action_type& action
    ) const
    {
        bool result;

        switch(action){
        case actions::up:
            result = state / width == 0;
            break;
        case actions::down:
            result = (state / width == height-2 && state % width > 0 && state % width < width-1)
                    || state / width == height-1;
            break;
        case actions::left:
            result = state % width == 0; // || state == height*width-1; <- is the goal condition
            break;
        case actions::right:
            result = state % width == width-1 || state == (height-1)*width;
            break;
        }

        return result;
    }

    feature_extractor fe;
    mutable std::default_random_engine gen;
};

#include <dlib/control/policy.h>
int main(int argc, char** argv)
{
    std::cout << "Hello." << std::endl;

    const auto height = 3u;
    const auto width = 5u;

    typedef cliff_model<height, width, feature_extractor> model_type;

    model_type model;
    qlearning<model_type> algorithm;
    algorithm.be_verbose();
    algorithm.set_max_iterations(100);

    auto policy = algorithm.train();

    auto s = model.initial_state();
    int r = 0; //TODO
    for(auto i = 0u; i < 100 && !model.is_final(s); i++){
        auto a = policy(s);
        auto new_s = model.step(s, a);
        r += model.reward(s,a,new_s);
        s = new_s;
    }

    if(!model.is_final(s))
        std::cout << "Nothing reached after 100 steps." << std::endl;
    else if(model.is_failure(s))
        std::cout << "Failed." << std::endl;
    else
        std::cout << "Success." << std::endl;

    std::cout << "Good bye." << std::endl;

    return 0;
}
