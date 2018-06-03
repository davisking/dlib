// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example showing how to use the dlib algorithms Q-learning and SARSA.
    These are two simples reinforcement learning algorithms. In short, they take a model
    and take steps over and over until they've learnt how to solve the given task properly.
*/

#include <dlib/matrix.h>
#include <dlib/control.h>
#include <dlib/serialize.h>
#include <limits>
#include <cmath>
#include <vector>
#include <iostream>

using namespace dlib;
using namespace std;

/*
    Both of these algorithms work with a reward system. It means that they assign to each
    pair (state, action) an expected reward (qvalue) and they iteratively update those values
    taking steps on an online model/simulation observing the reward obtained. Like so, they
    need a model class that allows them to work in a interactive way.

    The algorithms/agents' objective is to maximize the expected reward by taking the proper
    steps.
*/

/*
    Let me now introduce you the conceptual model the agent is going to use. Basically,
    this class represents a grid with a given height and width of the form
                                     ..........
                                     ..........
                                     IFFFFFFFFG
    where: - Fs represent pit cells where the agent can fall and thus fail the simulation.
           - I is the starting position.
           - G is the goal cell where the agent aims to go.
           - dots (.) represent free cells where the agent can freely go through.

    The agent receives the following reward: -100 for reaching F, 100 for reaching G and a
    reward of -1 otherwise.

    This model doesn't allow the agent to go out of bounds, instead it will stay in the same cell
    it was before the taking action (like if there was a wall) and receive a reward of -1.

    Function approximation by feature extraction is a powerful tool for reducing the state space's size.
    But ours is a toy example and so I use a one-shot representation, meaning that each feature
    represents a single state of the space and it will be 1 when that's state is active and 0 otherwise.
*/

//This is an implementation of the example_online_model interface on approximate_linear_models_abstract.h
template <
        int height,
        int width
        >
class cliff_model
{
public:
    // actions allowed in the model
    enum class actions {up = 0, right, down, left};
    constexpr static int num_actions = 4;

    // some constants we need
    constexpr static double EPS = 1e-16;
    constexpr static int HEIGHT = height;
    constexpr static int WIDTH = width;

    // we define the model's types
    typedef int state_type;
    typedef actions action_type;

    // Constructor
    explicit cliff_model(
        int seed = 0
    ) : gen(seed){}

    cliff_model(const cliff_model<height, width>&) = default;
    cliff_model<height, width>& operator=(const cliff_model<height, width>&) = default;

    // Functions that will use the agent

    unsigned int num_features(
    ) const { return num_actions * height * width; }

    void get_features(
        const state_type &state,
        const action_type &action,
        matrix<double,0,1>& feats
    ) const
    {
        feats.set_size(num_features());
        feats = 0;
        feats(num_actions*state + static_cast<int>(action)) = 1; //only this one is 1
    }

    // It's possible that the allowed actions differ among states.
    // In this case all movements are always allowed so we don't need to use state.
    action_type random_action(
        const state_type& state
    ) const
    {
        uniform_int_distribution<int> dist(0,num_actions-1);
        return static_cast<action_type>(dist(gen));
    }

    action_type find_best_action(
        const state_type& state,
        const matrix<double,0,1>& w
    ) const
    {
        auto best = numeric_limits<double>::lowest();
        std::vector<int> best_indexes;

        for(auto i = 0; i < num_actions; i++)
        {
            matrix<double,0,1> feats;
            get_features(state, static_cast<action_type>(i), feats);
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

    // This functions gives the rewards, that is, tells the agent how good are its movements
    double reward(
        const state_type &state,
        const action_type &action,
        const state_type &new_state
    ) const
    {
        return !is_final(new_state) ? -1 : is_success(new_state) ? 100 : -100;
    }

    state_type initial_state(
    ) const { return static_cast<state_type>((height-1) * width); }

    // This is an important function, basically it allows the agent to move around the environment
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

    // this functions allow the agent to know in which state of the simulation it's in
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

        switch(action)
        {
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

    // for accessing to gen on serialization functions (alternatively we could define a getter method)
    template < int H, int W> friend void serialize(const cliff_model<H, W>& item, std::ostream& out);
    template < int H, int W> friend void deserialize(cliff_model<H, W>& item, std::istream& in);

    mutable default_random_engine gen; //mutable because it doesn't changes the model state
};

template < int height, int width >
inline void serialize(const cliff_model<height, width>& item, std::ostream& out)
{
    int version = 1;
    dlib::serialize(version, out);
    dlib::serialize(item.gen, out);
}

template < int height, int width >
inline void deserialize(cliff_model<height, width>& item, std::istream& in)
{
    int version = 0;
    dlib::deserialize(version, in);
    if (version != 1)
        throw serialization_error("Unexpected version found while deserializing reinforcement learning test model object.");

    item = cliff_model<height, width>();
    dlib::deserialize(item.gen, in);
}

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

    The difference between executions comes in the way they train. Namely, the way they update the qvalue.
    Let's suppose that we are in the pair (s, a) and we are going to be in (s', a') in the next step.

    Q-learning is an off-policy algorithm meaning that doesn't consider its trully next move but the best one,
    that is, doesn't consider a'. Its update function is like this:
                            Q(s, a) = (1 - lr) * Q(s,a) + lr * (reward + disc * max_c Q(s', c))
    That formula means that it takes a convex combination of the current qvalue and the expected qvalue, but
    for doing so it considers the action c that maximizes Q(s', c) instead of the one he will take.

    On the other hand SARSA does exactly the same as Q-learning but it considers the action that he will do
    in the next step instead of the optimal. So it's an on-policy algorithm. Its update formula is:
                            Q(s, a) = (1 - lr) * Q(s,a) + lr * (reward + disc * Q(s', a'))

    This looks as a meaningless change, but what produces is that, when training, SARSA tends to be more conservative
    in its movements while Q-learning tries to optimizes them no matter what. In cases when you have to avoid failure
    (usually a real world example) SARSA is a better option.

    In our example this difference can be appreciated in the way they learn. Q-learning will try to go close to the pit
    cells all the time (falling a lot in the training process) and SARSA will go one or two cells off the cliff.
*/
template <
        typename model_t,
        typename algorithm_t // qlearning or sarsa
        >
void run_example(const model_t &model, algorithm_t &&algorithm)
{
    //algorithm.be_verbose();  // uncomment it if you want to see some training info.
    auto policy = algorithm.train(model);

    cout << "Starting final simulation..." << endl;
    auto s = model.initial_state();
    double r = 0.;
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
    cliff_model<height, width> model;

    char response;
    cout << "Qlearning or SARSA? (q/s): ";
    cin >> response;

    if(response == 'q')
        run_example(model, qlearning<decltype(model)>());
    else if(response == 's')
        run_example(model, sarsa<decltype(model)>());
    else
        cerr << "Invalid option." << endl;

    cout << "Good bye." << endl;

    return 0;
}
