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

    template <
        typename T
        >
    struct constituent 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents the linguistic idea of a constituent, that is, a
                group of words that functions as a single unit.  In particular, it
                represents a combination of two constituents into a new constituent.

                Additionally, a constituent object represents a range of words relative to
                some std::vector of words.  The range is from [begin, end) (i.e. including
                begin but not including end, so using the normal C++ iterator notation).
                Moreover, a constituent is always composed of two parts, each having a tag.
                Therefore, the left part is composed of the words in the range [begin,k)
                and has tag left_tag while the right part of the constituent contains the
                words in the range [k,end) and has the tag right_tag.

                The tags are user defined objects of type T.  In general, they are used to
                represent syntactic categories such as noun phrase, verb phrase, etc.
        !*/

        unsigned long begin, end, k;
        T left_tag; 
        T right_tag;
    };

    template <
        typename T
        >
    void serialize(
        const constituent<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support
    !*/

    template <
        typename T
        >
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

    template <
        typename T
        >
    struct parse_tree_element
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is used to represent a node in a binary parse tree.  An entire
                parse tree is represented by a std::vector of parse_tree_element objects.
                We follow the convention that the first element of this vector is always
                the root of the entire tree.

                The fields of this object have the following interpretations:   
                    - c == the constituent spanned by this node in the parse tree.
                      Therefore, the node spans the words in the range [c.begin, c.end).
                    - tag == the syntactic category of this node in the parse tree.
                    - score == the score or log likelihood for this parse tree.  In
                      general, this is the sum of scores of all the production rules used
                      to build the tree rooted at the current node.
                    - let PT denote the vector of parse_tree_elements that defines an
                      entire parse tree.  Then we have:
                        - if (left != END_OF_TREE) then
                            - PT[left] == the left sub-tree of the current node.
                            - PT[left] spans the words [c.begin, c.k)
                            - PT[left].tag == c.left_tag
                        - else
                            - there is no left sub-tree

                        - if (right != END_OF_TREE) then
                            - PT[right] == the right sub-tree of the current node.
                            - PT[right] spans the words [c.k, c.end)
                            - PT[right].tag == c.right_tag
                        - else
                            - there is no right sub-tree
        !*/

        constituent<T> c;
        T tag; 
        double score; 

        unsigned long left;
        unsigned long right; 
    };

    template <
        typename T
        >
    void serialize (
        const parse_tree_element<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support
    !*/

    template <
        typename T
        >
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
        const std::vector<T>& words,
        const constituent<T>& c,
        std::vector<std::pair<T,double> >& possible_tags
    )
    /*!
        requires
            - 0 <= c.begin < c.k < c.end <= words.size()
            - possible_tags.size() == 0 
        ensures
            - Finds all the syntactic categories that can be used to label c and puts those
              categories, along with their scores, into possible_tags.  Or in other words,
              this function determines which production rules can be used to turn the left
              and right sub-constituents in c into a single constituent.  The contents of c
              have the following interpretations:
                - The left sub-constituent has syntactic category c.left_tag 
                - for all i such that c.begin <= i < c.k: 
                    - words[i] is part of the left sub-constituent.
                - The right sub-constituent has syntactic category c.right_tag 
                - for all i such that c.k <= i < c.end: 
                    - words[i] is part of the right sub-constituent.

            - Note that example_production_rule_function() is not a real function.  It is
              here just to show you how to define production rule producing functions for
              use with the find_max_parse_cky() routine defined below.
    !*/

    template <
        typename T, 
        typename production_rule_function
        >
    void find_max_parse_cky (
        const std::vector<T>& words,
        const production_rule_function& production_rules,
        std::vector<std::vector<parse_tree_element<T> > >& parse_trees
    );
    /*!
        requires
            - production_rule_function == a function or function object with the same
              interface as example_production_rule_function defined above.
        ensures
            - Uses the CKY algorithm to find the most probable/highest scoring parse tree
              of the given vector of words.  The output is stored in #parse_trees.
            - This function outputs a set of non-overlapping parse trees.  Each parse tree
              always spans the largest number of words possible, regardless of any other
              considerations (except that the parse trees cannot have overlapping word
              spans).  For example, this function will never select a smaller parse tree,
              even if it would have a better score, if it can possibly build a larger tree.
              Therefore, this function will only output multiple parse trees if it is
              impossible to form words into a single parse tree.
            - This function uses production_rules() to find out what the allowed production
              rules are.  That is, production_rules() defines all properties of the grammar
              used by find_max_parse_cky(). 
            - for all valid i:
                - #parse_trees[i].size() != 0
                - #parse_trees[i] == the root of the i'th parse tree.
                - #parse_trees[i].score == the score of the i'th parse tree. 
                - The i'th parse tree spans all the elements of words in the range
                  [#parse_trees[i].c.begin, #parse_trees[i].c.end).
    !*/

// -----------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------

    class parse_tree_to_string_error : public error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is the exception thrown by parse_tree_to_string() and
                parse_tree_to_string_tagged() if the inputs are discovered to be invalid.
        !*/
    };

// -----------------------------------------------------------------------------------------

    template <
        typename T, 
        typename U
        >
    std::string parse_tree_to_string (
        const std::vector<parse_tree_element<T> >& tree,
        const std::vector<U>& words
    );
    /*!
        requires
            - It must be possible to print U objects to an ostream using operator<<
              (typically, U would be something like std::string)
        ensures
            - Interprets tree as a parse tree defined over the given sequence of words.  
            - returns a bracketed string that represents the parse tree over the words.  
              For example, suppose the following parse tree is input:

                        /\
                       /  \
                      /\   \
                     /  \   \
                   the dog  ran

              Then the output would be the string "[[the dog] ran]"
        throws
            - parse_tree_to_string_error
                This exception is thrown if an invalid tree is detected.  This might happen
                if the tree refers to elements of words that don't exist because words is
                shorted than it is supposed to be.
    !*/

// -----------------------------------------------------------------------------------------

    template <
        typename T, 
        typename U
        >
    std::string parse_tree_to_string_tagged (
        const std::vector<parse_tree_element<T> >& tree,
        const std::vector<U>& words
    );
    /*!
        requires
            - It must be possible to print T objects to an ostream using operator<<
            - It must be possible to print U objects to an ostream using operator<<
              (typically, U would be something like std::string)
        ensures
            - This function does the same thing as parse_tree_to_string() except that it
              also includes the parse_tree_element::tag object in the output.  Therefore,
              the tag of each bracket will be included as the first token inside the
              bracket.  For example, suppose the following parse tree is input (where tags
              are shown at the vertices):

                        S
                        /\
                      NP  \
                      /\   \
                     /  \   \
                   the dog  ran

              Then the output would be the string "[S [NP the dog] ran]"
        throws
            - parse_tree_to_string_error
                This exception is thrown if an invalid tree is detected.  This might happen
                if the tree refers to elements of words that don't exist because words is
                shorted than it is supposed to be.
    !*/

// -----------------------------------------------------------------------------------------

}

#endif // DLIB_FIND_MAX_PARsE_CKY_ABSTRACT_H__

