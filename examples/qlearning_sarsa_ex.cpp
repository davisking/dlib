// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example showing how to use the dlib algorithms Q-learning and SARSA.
    These are two simples reinforcement learning algorithms. In short, they take a model
    and take steps over and over until they've learnt how to solve the given task properly.
*/

#include <dlib/matrix.h>
#include <dlib/control.h>
#include <limits>
#include <cmath>
#include <vector>
#include <iostream>

using namespace dlib;
using namespace std;

/*
    Both of these algorithms work by a reward system. That means that they assign to each
    pair (state, action) an expected reward (Qvalue) and they update those values iteratively
    taking steps on a model/simulation and observing the reward they obtain. Like so, they
    need a model class that allow them to work in a interactive way.

    The algorithms/agents objective is to maximize the expected reward by taking the proper
    steps.
*/

/*
    This is the model the agent is going to work with in the example. In particular,
    this class represents a grid with a given height and width of the form
                                     ..........
                                     ..........
                                     IFFFFFFFFG
    where: - F are pit cells (if the agent falls there it fails and simulation ends).
           - I is the starting position.
           - G is the goal cell (the agent goal is to reach that cell).
           - . are free cells where the agent can go.

    The agent receives the following reward: -100 for reaching F, 100 for reaching G and a
    reward of -1 otherwise.

    This model doesn't allow the agent to go out of bounds, instead it will stay in the same cell
    he was before the action (like if there was a wall there) but receiving a reward of -1.
*/
template <
        int height,
        int width,
        template<typename,typename> class feature_extractor_type
        >
class cliff_model
{
public:
    // actions allowed in the model
    enum class actions {up = 0, right, down, left};
    constexpr static int num_actions = 4;

    // some constants that we need
    constexpr static double EPS = 1e-16;
    constexpr static int HEIGHT = height;
    constexpr static int WIDTH = width;

    // we define the model's types
    typedef int state_type;
    typedef actions action_type;
    typedef int reward_type;

    // this ensures that the feature extractor uses the same underlying types as our model
    typedef feature_extractor_type<state_type, action_type> feature_extractor;


    // Constructor
    explicit cliff_model(
        int seed = 0
    ) : fe(height, width, num_actions), gen(seed){}


    // Functions that will use the agent

    // It returns a random action. It's possible that the allowed actions differ from among states.
    // In this case all movements are always allowed so we don't need to use state.
    action_type random_action(
        const state_type& state
    ) const
    {
        uniform_int_distribution<int> dist(0,num_actions-1);
        return static_cast<action_type>(dist(gen));
    }

    // Returns the best action that maximizes the expected reward, that is,
    // the action that maximizes dot_product(w, get_features(state, action))
    // w will be the weights assign by the agent to each feature
    action_type find_best_action(
        const state_type& state,
        const matrix<double,0,1>& w
    ) const
    {
        auto best = numeric_limits<double>::lowest();
        auto best_indexes = std::vector<int>();

        for(auto i = 0; i < num_actions; i++){
            auto feats = get_features(state, static_cast<action_type>(i));
            auto product = dot(w, feats);

            if(product > best){
                best = product;
                best_indexes.clear();
            }
            if(abs(product - best) < EPS)
                best_indexes.push_back(i);
        }

        // returns a random action between the best ones.
        uniform_int_distribution<unsigned long> dist(0, best_indexes.size()-1);
        return static_cast<action_type>(best_indexes[dist(gen)]);
    }


    // This functions are delegated to the feature extractor

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


    // This functions gives the rewards, that is, tells the agent how good are its movements
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

    // This is an important function, basically it allows the agent to move in the model's world
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

    // this functions allow the agent to know in which state of the simulation he is in

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
            result = state / width == height-1;
            break;
        case actions::left:
            result = state % width == 0;
            break;
        case actions::right:
            result = state % width == width-1;
            break;
        }

        return result;
    }

    feature_extractor fe;
    mutable default_random_engine gen; //mutable because it doesn't changes the model state
};

/*
    Usually when we use these types of agents the state space of the model is huge. That could make
    the Qfunction to be unmanageable and so we need to use what is known as function approximation.

    Basically it represents the states by a given features instead of the states themselves. That way
    what usually was just a single value Q(state, action) now is codified as the linear combination of
    learnt weights and the features, that is, Q(state, action) = dot_product(weights, features(state, action)).

    Our example is a toy example and so we don't need to use it. However, to show how it works I use a simple
    one-shot representation of the states. That means that I have a vector of features where the feature in the
    ith position is one if we provide a specific (state, action) and 0 otherwise.
*/
template <
        typename state_type,
        typename action_type
        >
class feature_extractor
{
public:
    feature_extractor(
        int h,
        int w,
        int na
    ) : height(h), width(w), num_actions(na) {}

    //the size of the vector
    inline long num_features(
    ) const { return num_actions * height * width; }

    matrix<double,0,1> get_features(
        const state_type &state,
        const action_type &action
    ) const
    {
        matrix<double,0,1> feats(num_features());
        feats = 0;
        feats(num_actions*state + static_cast<int>(action)) = 1; //only this one is 1

        return feats;
    }

private:
    int height, width, num_actions;
};

// This is just a helper function to pretty-print the agent's state.
template <
    typename model_t
    >
void print(
        ostream &os,
        const model_t &model,
        const typename model_t::state_type &state,
        const matrix<double,0,1> &weights,
        const typename model_t::action_type &action
)
{
    cout << "weights: ";
    for(int i = 0; i < 4; i++)
        cout << weights(state*4+i) << " ";
    cout << endl;

    cout << "action: " << static_cast<int>(action) << "\n";

    for(auto i = 0; i < model_t::HEIGHT; i++){
        for(auto j = 0; j < model_t::WIDTH; j++){
            typename model_t::state_type s = model_t::WIDTH * i + j;
            os << ( s == state ? 'X' : model.is_success(s) ? 'G' : model.is_failure(s) ? 'F' : '.');
        }
        os << endl;
    }
    os << endl;
}

/*
    This is the function that runs the agent. The code to run both agents are identical so I
    chose to use a templated function.

    The difference between executions comes in the way they train. Namely, the way they updated the Qvalue.
    Let's suppose that we are in the pair (s, a) and we are going to be in (s', a') in the next step.

    Q-learning is an off-policy algorithm meaning that doesn't consider its trully next move but the best one,
    that is, doesn't consider a'. Its update function is like this:
                            Q(s, a) = (1 - lr) * Q(s,a) + lr * (reward + disc * max_c Q(s', c))
    That formula means that it takes a convex combination of the current qvalue and the expected qvalue, but
    for doing so it considers the action c that maximizes Q(s', c) instead of the one he will take.

    On the other hand SARSA does exactly the same as Q-learning but it considers the action that he will do
    in the next step instead of the optimal. So it's an on-policy algorithm. Its update formula is:
                            Q(s, a) = (1 - lr) * Q(s,a) + lr * (reward + disc * Q(s', a'))

    This seems as a meaningless change, but what produces is that when training SARSA tends to be more conservative
    in its movement while Q-learning tries to optimizes no matter what. In cases when you have to avoid failling
    (usually a real world example) SARSA is a better option.

    In our example this difference is appreciated in the way they learn. Q-learning will try to go close to the pit
    cells all the time (falling a lot in the training process) and SARSA will go one or two cells off the cliff.

    Usually, one decreases the learning ratio as the iterations go on and so SARSA would converge to the same solution
    as Q-learning. This is not implemented yet and so the learning rate is constant always.
*/
template <
        typename model_t,
        typename algorithm_t // this can be qlearning or sarsa
        >
void run_example(const model_t &model, algorithm_t &&algorithm)
{
    //algorithm.be_verbose();  // uncomment it if you want to see some training info.
    auto policy = algorithm.train(model);

    cout << "Starting final simulation..." << endl;
    auto s = model.initial_state();
    auto r = static_cast<typename model_t::reward_type>(0);
    int i;

    for(i = 0; i < 100 && !model.is_final(s); i++){
        auto a = policy(s);
        auto new_s = model.step(s, a);
        r += model.reward(s,a,new_s);

        print(cout, model, s, policy.get_weights(), a);
        s = new_s;
    }
    print(cout, model, s, policy.get_weights(), static_cast<decltype(policy(s))>(0));
    cout << "Simulation finished." << endl;

    if(!model.is_final(s))
        cout << "Nothing reached after 100 steps." << endl;
    else if(model.is_failure(s))
        cout << "Failed after " << i << " steps with reward " << r << "." << endl;
    else
        cout << "Success after " << i << " steps with reward " << r << "." << endl;
}

int main(int argc, char** argv)
{
    cout << "Hello." << endl;

    const auto height = 3u;
    const auto width = 7u;
    typedef cliff_model<height, width, feature_extractor> model_type;
    model_type model;

    char response;
    cout << "Qlearning or SARSA? (q/s): ";
    cin >> response;

    if(response == 'q')
        run_example(model, qlearning());
    else if(response == 's')
        run_example(model, sarsa());
    else
        cerr << "Invalid option." << endl;

    cout << "Good bye." << endl;

    return 0;
}
