// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FIND_MAX_PARsE_CKY_ABSTRACT_H__
#ifdef DLIB_FIND_MAX_PARsE_CKY_ABSTRACT_H__

#include <vector>
#include <string>
#include "../algs.h" 

namespace dlib
{

// -----------------------------------------------------------------------------------------

    template <typename T>
    struct constituent 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

        unsigned long begin, end, k;
        T left_tag; 
        T right_tag;
    };

    template <typename T>
    void serialize(
        const constituent<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support
    !*/

    template <typename T>
    void deserialize(
        constituent<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support 
    !*/

// -----------------------------------------------------------------------------------------

    /*!A END_OF_TREE is used to indicate that parse_tree_element::left or
         parse_tree_element::right doesn't point to another subtree.
    !*/
    const unsigned long END_OF_TREE = 0xFFFFFFFF;

// -----------------------------------------------------------------------------------------

    template <typename T>
    struct parse_tree_element
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

        constituent<T> c;
        T tag; // id for the constituent corresponding to this level of the tree

        // subtrees.  These are the index values into the std::vector that contains all the parse_tree_elements.
        unsigned long left;
        unsigned long right; 

        double score; // score for this tree
    };

    template <typename T>
    void serialize (
        const parse_tree_element<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support
    !*/

    template <typename T>
    void deserialize (
        parse_tree_element<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support 
    !*/

// -----------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------

    void example_production_rule_function (
        const std::vector<T>& sequence,
        const constituent<T>& c,
        std::vector<std::pair<T,double> >& possible_tags
    )
    /*!
        requires
            - 0 <= c.begin < c.k < c.end <= sequence.size()
            - possible_tags.size() == 0 
        ensures
            - finds all the production rules that can turn c into a single non-terminal.
              Puts the IDs of these rules and their scores into possible_tags.
            - Note that example_production_rule_function() is not a real function.  It is
              here just to show you how to define production rule producing functions
              for use with the find_max_parse_cky() routine defined below.
    !*/

    template <
        typename T, 
        typename production_rule_function
        >
    void find_max_parse_cky (
        const std::vector<T>& sequence,
        const production_rule_function& production_rules,
        std::vector<std::vector<parse_tree_element<T> > >& parse_trees
    );
    /*!
        requires
            - production_rule_function == a function or function object with the same
              interface as example_production_rule_function defined above.
    !*/

// -----------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------

    class parse_tree_to_string_error : public error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/
    };

// -----------------------------------------------------------------------------------------

    template <typename T, typename U>
    std::string parse_tree_to_string (
        const std::vector<parse_tree_element<T> >& tree,
        const std::vector<U>& items
    );
    /*!
        ensures
            - 
    !*/

// -----------------------------------------------------------------------------------------

    template <typename T, typename U>
    std::string parse_tree_to_string_tagged (
        const std::vector<parse_tree_element<T> >& tree,
        const std::vector<U>& items
    );
    /*!
        ensures
            - 
    !*/

// -----------------------------------------------------------------------------------------

}

#endif // DLIB_FIND_MAX_PARsE_CKY_ABSTRACT_H__

