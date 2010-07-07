
#include "to_xml.h"
#include "dlib/dir_nav.h"
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stack>
#include "dlib/cpp_tokenizer.h"
#include "dlib/string.h"

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

typedef cpp_tokenizer::kernel_1a_c tok_type;

// ----------------------------------------------------------------------------------------

class file_filter
{
public:

    file_filter( 
        const string& filter
    )
    {
        // pick out the filter strings
        istringstream sin(filter);
        string temp;
        sin >> temp;
        while (sin)
        {
            endings.push_back("." + temp);
            sin >> temp;
        }
    }

    bool operator() ( const file& f) const
    {
        // check if any of the endings match
        for (unsigned long i = 0; i < endings.size(); ++i)
        {
            // if the ending is bigger than f's name then it obviously doesn't match
            if (endings[i].size() > f.name().size())
                continue;

            // now check if the actual characters that make up the end of the file name 
            // matches what is in endings[i].
            if ( std::equal(endings[i].begin(), endings[i].end(), f.name().end()-endings[i].size()))
                return true;
        }

        return false;
    }

    std::vector<string> endings;
};

// ----------------------------------------------------------------------------------------

void obtain_list_of_files (
    const cmd_line_parser<char>::check_1a_c& parser, 
    const std::string& filter, 
    const unsigned long search_depth,
    std::vector<std::pair<string,string> >& files
)
{
    for (unsigned long i = 0; i < parser.option("i").count(); ++i)
    {
        const directory dir(parser.option("i").argument(0,i));

        const std::vector<file>& temp = get_files_in_directory_tree(dir, file_filter(filter), search_depth);

        // figure out how many characters need to be removed from the path of each file
        const string parent = dir.get_parent().full_name();
        unsigned long strip = parent.size();
        if (parent.size() > 0 && parent[parent.size()-1] != '\\' && parent[parent.size()-1] != '/')
            strip += 1;

        for (unsigned long i = 0; i < temp.size(); ++i)
        {
            files.push_back(make_pair(temp[i].full_name().substr(strip), temp[i].full_name()));
        }
    }

    for (unsigned long i = 0; i < parser.number_of_arguments(); ++i)
    {
        files.push_back(make_pair(parser[i], parser[i]));
    }

    std::sort(files.begin(), files.end());
}

// ----------------------------------------------------------------------------------------

struct tok_function_record
{
    std::vector<std::pair<int,string> > declaration;
    string scope;
    string file;
    string comment;
};

struct tok_method_record
{
    std::vector<std::pair<int,string> > declaration;
    string comment;
};

struct tok_variable_record
{
    std::vector<std::pair<int,string> > declaration;
};

struct tok_typedef_record
{
    std::vector<std::pair<int,string> > declaration;
};

struct tok_class_record 
{
    std::vector<std::pair<int,string> > declaration;
    string name;
    string scope;
    string file;
    string comment;

    std::vector<tok_method_record> public_methods;
    std::vector<tok_variable_record> public_variables;
    std::vector<tok_typedef_record> public_typedefs;
    std::vector<tok_class_record> public_inner_classes;
};

// ----------------------------------------------------------------------------------------

struct function_record
{
    string name;
    string scope;
    string declaration;
    string file;
    string comment;
};

struct method_record
{
    string name;
    string declaration;
    string comment;
};

struct variable_record
{
    string declaration;
};

struct typedef_record
{
    string declaration;
};

struct class_record 
{
    string name;
    string scope;
    string declaration;
    string file;
    string comment;

    std::vector<method_record> public_methods;
    std::vector<variable_record> public_variables;
    std::vector<typedef_record> public_typedefs;
    std::vector<class_record> public_inner_classes;
};

// ----------------------------------------------------------------------------------------

unsigned long count_newlines (
    const string& str
)
/*!
    ensures
        - returns the number of '\n' characters inside str
!*/
{
    unsigned long count = 0;
    for (unsigned long i = 0; i < str.size(); ++i)
    {
        if (str[i] == '\n')
            ++count;
    }
    return count;
}

// ----------------------------------------------------------------------------------------

bool contains_unescaped_newline (
    const string& str
)
/*!
    ensures
        - returns true if str contains a '\n' character that isn't preceded by a '\' 
          character.
!*/
{
    if (str.size() == 0)
        return false;

    if (str[0] == '\n')
        return true;

    for (unsigned long i = 1; i < str.size(); ++i)
    {
        if (str[i] == '\n' && str[i-1] != '\\')
            return true;
    }

    return false;
}

// ----------------------------------------------------------------------------------------

bool is_formal_comment (
    const string& str
)
{
    if (str.size() < 6)
        return false;

    if (str[0] == '/' &&
        str[1] == '*' &&
        str[2] == '!' &&
        str[3] != 'P' &&
        str[3] != 'p' &&
        str[str.size()-3] == '!' &&
        str[str.size()-2] == '*' &&
        str[str.size()-1] == '/' )
        return true;

    return false;
}

// ----------------------------------------------------------------------------------------

string make_scope_string (
    const std::vector<string>& namespaces,
    unsigned long exclude_last_num_scopes = 0 
)
{
    string temp;
    for (unsigned long i = 0; i + exclude_last_num_scopes < namespaces.size(); ++i)
    {
        if (namespaces[i].size() == 0)
            continue;

        if (temp.size() == 0)
            temp = namespaces[i];
        else
            temp += "::" + namespaces[i];
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

bool looks_like_function_declaration (
    const std::vector<std::pair<int,string> >& declaration
)
{

    // Check if declaration contains IDENTIFIER ( ) somewhere in it.
    bool seen_first_part = false;
    bool seen_operator = false;
    int local_paren_count = 0;
    for (unsigned long i = 1; i < declaration.size(); ++i)
    {
        if (declaration[i].first == tok_type::KEYWORD &&
            declaration[i].second == "operator")
        {
            seen_operator = true;
        }

        if (declaration[i].first == tok_type::OTHER &&
            declaration[i].second == "(" &&
            (declaration[i-1].first == tok_type::IDENTIFIER || seen_operator))
        {
            seen_first_part = true;
        }

        if (declaration[i].first == tok_type::OTHER) 
        {
            if ( declaration[i].second == "(")
                ++local_paren_count;
            else if ( declaration[i].second == ")")
                --local_paren_count;
        }
    }

    if (seen_first_part && local_paren_count == 0)
        return true;
    else
        return false;
}

// ----------------------------------------------------------------------------------------

void process_file (
    istream& fin,
    const string& file,
    std::vector<tok_function_record>& functions,
    std::vector<tok_class_record>& classes
)
/*!
    ensures
        - scans the given file for global functions and appends any found into functions.
        - scans the given file for global classes and appends any found into classes.
!*/
{
    tok_type tok;
    tok.set_stream(fin);

    bool recently_seen_struct_keyword = false;
        // true if we have seen the struct keyword and
        // we have not seen any identifiers or { characters

    string last_struct_name;
        // the name of the last struct we have seen

    bool recently_seen_class_keyword = false;
        // true if we have seen the class keyword and
        // we have not seen any identifiers or { characters

    string last_class_name;
        // the name of the last class we have seen

    bool recently_seen_namespace_keyword = false;
        // true if we have seen the namespace keyword and
        // we have not seen any identifiers or { characters

    string last_namespace_name;
        // the name of the last namespace we have seen

    bool recently_seen_pound_define = false;
        // true if we have seen a #define and haven't seen an unescaped newline

    bool recently_seen_typedef = false;
        // true if we have seen a typedef keyword and haven't seen a ;

    bool recently_seen_paren_0 = false;
        // true if we have seen paren_count transition to zero but haven't yet seen a ; or { or 
        // a new line if recently_seen_pound_define is true.

    bool recently_seen_closing_bracket = false;
        // true if we have seen a } and haven't yet seen an IDENTIFIER or ;

    bool recently_seen_new_scope = false;  
        // true if we have seen the keywords class, namespace, struct, or extern and
        // we have not seen the characters {, ), or ; since then

    bool at_top_of_new_scope = false;
        // true if we have seen the { that started a new scope but haven't seen anything yet but WHITE_SPACE

    std::vector<string> namespaces; 
        // a stack to hold the names of the scopes we have entered.  This is the classes, structs, and namespaces we enter.
    namespaces.push_back(""); // this is the global namespace


    std::stack<bool> inside_public_scope;
        // If the stack isn't empty then we are inside a class or struct and the top value
        // in the stack tells if we are in a public region.

    std::stack<unsigned long> scopes; // a stack to hold current and old scope counts 
                             // the top of the stack counts the number of new scopes (i.e. unmatched { ) we have entered 
                             // since we were at a scope where functions can be defined.
                             // We also maintain the invariant that scopes.size() == namespaces.size()
    scopes.push(0);

    std::stack<tok_class_record> class_stack;
        // This is a stack where class_stack.top() == the incomplete class record for the class declaration we are
        // currently in.

    unsigned long paren_count = 0; 
        // this is the number of ( we have seen minus the number of ) we have
        // seen.

    std::vector<std::pair<int,string> > token_accum;
        // Used to accumulate tokens for function and class declarations

    std::vector<std::pair<int,string> > last_full_declaration;
        // Once we determine that token_accum has a full declaration in it we copy it into last_full_declaration. 

    int type;
    string token;

    tok.get_token(type, token);

    while (type != tok_type::END_OF_FILE)
    {
        switch(type)
        {
            case tok_type::KEYWORD: // ------------------------------------------
                {
                    token_accum.push_back(make_pair(type,token));

                    if (token == "class")
                    {
                        recently_seen_class_keyword = true;
                        recently_seen_new_scope = true;
                    }
                    else if (token == "struct")
                    {
                        recently_seen_struct_keyword = true;
                        recently_seen_new_scope = true;
                    }
                    else if (token == "namespace")
                    {
                        recently_seen_namespace_keyword = true;
                        recently_seen_new_scope = true;
                    }
                    else if (token == "extern")
                    {
                        recently_seen_new_scope = true;
                    }
                    else if (token == "#define")
                    {
                        recently_seen_pound_define = true;
                    }
                    else if (token == "typedef")
                    {
                        recently_seen_typedef = true;
                    }
                    else if (recently_seen_pound_define == false)
                    {
                        // eat white space
                        int temp_type;
                        string temp_token;
                        if (tok.peek_type() == tok_type::WHITE_SPACE)
                            tok.get_token(temp_type, temp_token);

                        const bool next_is_colon = (tok.peek_type() == tok_type::OTHER && tok.peek_token() == ":");
                        if (next_is_colon)
                        {
                            // eat the colon
                            tok.get_token(temp_type, temp_token);

                            if (inside_public_scope.size() > 0 && token == "public")
                            {
                                inside_public_scope.top() = true;
                                token_accum.clear();
                                last_full_declaration.clear();
                            }
                            else if (inside_public_scope.size() > 0 && token == "protected")
                            {
                                inside_public_scope.top() = true;
                                token_accum.clear();
                                last_full_declaration.clear();
                            }
                            else if (inside_public_scope.size() > 0 && token == "private")
                            {
                                inside_public_scope.top() = false;
                                token_accum.clear();
                                last_full_declaration.clear();
                            }
                        }
                    }

                    at_top_of_new_scope = false;

                }break;

            case tok_type::COMMENT: // ------------------------------------------
                {
                    if (scopes.top() == 0 && last_full_declaration.size() > 0 && is_formal_comment(token) &&
                        paren_count == 0)
                    {

                        // if we are inside a class or struct
                        if (inside_public_scope.size() > 0)
                        {
                            // if we are looking at a comment at the top of a class
                            if (at_top_of_new_scope)
                            {
                                // push an entry for this class into the class_stack
                                tok_class_record temp;
                                temp.declaration = last_full_declaration;
                                temp.file = file;
                                temp.name = namespaces.back();
                                temp.scope = make_scope_string(namespaces,1);
                                temp.comment = token;
                                class_stack.push(temp);
                            }
                            else if (inside_public_scope.top())
                            {
                                // This should be a member function.  
                                // Only do anything if the class that contains this member function is
                                // in the class_stack.
                                if (class_stack.size() > 0 && class_stack.top().name == namespaces.back() &&
                                    looks_like_function_declaration(last_full_declaration))
                                {
                                    tok_method_record temp;
                                    temp.declaration = last_full_declaration;
                                    temp.comment = token;
                                    class_stack.top().public_methods.push_back(temp);
                                }
                            }
                        }
                        else
                        {
                            // we should be looking at a global declaration of some kind.   
                            if (looks_like_function_declaration(last_full_declaration))
                            {
                                tok_function_record temp;
                                temp.declaration = last_full_declaration;
                                temp.file = file;
                                temp.scope = make_scope_string(namespaces);
                                temp.comment = token;
                                functions.push_back(temp);
                            }
                        }

                        token_accum.clear();
                        last_full_declaration.clear();
                    }

                    at_top_of_new_scope = false;
                }break;

            case tok_type::IDENTIFIER: // ------------------------------------------
                {
                    if (recently_seen_class_keyword)
                    {
                        last_class_name = token;
                        last_struct_name.clear();
                        last_namespace_name.clear();
                    }
                    else if (recently_seen_struct_keyword)
                    {
                        last_struct_name = token;
                        last_class_name.clear();
                        last_namespace_name.clear();
                    }
                    else if (recently_seen_namespace_keyword)
                    {
                        last_namespace_name = token;
                        last_class_name.clear();
                        last_struct_name.clear();
                    }

                    recently_seen_class_keyword = false;
                    recently_seen_struct_keyword = false;
                    recently_seen_namespace_keyword = false;
                    recently_seen_closing_bracket = false;
                    at_top_of_new_scope = false;

                    token_accum.push_back(make_pair(type,token));
                }break;

            case tok_type::OTHER: // ------------------------------------------
                {
                    switch(token[0])
                    {
                        case '{':
                            // if we are entering a new scope
                            if (recently_seen_new_scope)
                            {
                                scopes.push(0);
                                at_top_of_new_scope = true;

                                // if we are entering a class 
                                if (last_class_name.size() > 0)
                                {
                                    inside_public_scope.push(false);
                                    namespaces.push_back(last_class_name);
                                }
                                else if (last_struct_name.size() > 0)
                                {
                                    inside_public_scope.push(true);
                                    namespaces.push_back(last_struct_name);
                                }
                                else if (last_namespace_name.size() > 0)
                                {
                                    namespaces.push_back(last_namespace_name);
                                }
                                else
                                {
                                    namespaces.push_back("");
                                }
                            }
                            else
                            {
                                scopes.top() += 1;
                            }
                            recently_seen_new_scope = false;
                            recently_seen_class_keyword = false;
                            recently_seen_struct_keyword = false;
                            recently_seen_namespace_keyword = false;
                            recently_seen_paren_0 = false;

                            // a { at function scope is an end of a potential declaration
                            if (scopes.top() == 0)
                            {
                                // put token_accum into last_full_declaration
                                token_accum.swap(last_full_declaration);
                            }
                            token_accum.clear();
                            break;

                        case '}':
                            if (scopes.top() > 0)
                            {
                                scopes.top() -= 1;
                            }
                            else if (scopes.size() > 1)
                            {
                                scopes.pop();
                                namespaces.pop_back();

                                if (inside_public_scope.size() > 0)
                                    inside_public_scope.pop();

                                // if this class is a inner_class of another then push it into the
                                // public_inner_classes field of it's containing class
                                if (class_stack.size() > 1)
                                {
                                    tok_class_record temp = class_stack.top();
                                    class_stack.pop();
                                    class_stack.top().public_inner_classes.push_back(temp);
                                }
                                else if (class_stack.size() > 0)
                                {
                                    classes.push_back(class_stack.top());
                                    class_stack.pop();
                                }
                            }

                            token_accum.clear();
                            recently_seen_closing_bracket = true;
                            at_top_of_new_scope = false;
                            break;

                        case ';':
                            // a ; at function scope is an end of a potential declaration
                            if (scopes.top() == 0)
                            {
                                // put token_accum into last_full_declaration
                                token_accum.swap(last_full_declaration);
                            }
                            token_accum.clear();

                            // if we are inside the public area of a class and this ; might be the end
                            // of a typedef or variable declaration
                            if (scopes.top() == 0 && inside_public_scope.size() > 0 && 
                                inside_public_scope.top() == true &&
                                recently_seen_closing_bracket == false)
                            {
                                if (recently_seen_typedef)
                                {
                                    // This should be a typedef inside the public area of a class or struct:
                                    // Only do anything if the class that contains this typedef is in the class_stack.
                                    if (class_stack.size() > 0 && class_stack.top().name == namespaces.back())
                                    {
                                        tok_typedef_record temp;
                                        temp.declaration = last_full_declaration;
                                        class_stack.top().public_typedefs.push_back(temp);
                                    }

                                }
                                else if (recently_seen_paren_0 == false && recently_seen_new_scope == false)
                                {
                                    // This should be some kind of public variable declaration inside a class or struct:
                                    // Only do anything if the class that contains this member variable is in the class_stack.
                                    if (class_stack.size() > 0 && class_stack.top().name == namespaces.back())
                                    {
                                        tok_variable_record temp;
                                        temp.declaration = last_full_declaration;
                                        class_stack.top().public_variables.push_back(temp);
                                    }

                                }
                            }

                            recently_seen_new_scope = false;
                            recently_seen_typedef = false;
                            recently_seen_paren_0 = false;
                            recently_seen_closing_bracket = false;
                            at_top_of_new_scope = false;
                            break;

                        case '(':
                            ++paren_count;
                            token_accum.push_back(make_pair(type,token));
                            at_top_of_new_scope = false;
                            break;

                        case ')':
                            token_accum.push_back(make_pair(type,token));

                            --paren_count;
                            if (paren_count == 0)
                            {
                                recently_seen_paren_0 = true;
                                if (scopes.top() == 0)
                                {
                                    last_full_declaration = token_accum;
                                }
                            }

                            recently_seen_new_scope = false;
                            at_top_of_new_scope = false;
                            break;

                        default:
                            token_accum.push_back(make_pair(type,token));
                            at_top_of_new_scope = false;
                            break;
                    }
                }break;


            case tok_type::WHITE_SPACE: // ------------------------------------------
                {
                    if (recently_seen_pound_define)
                    {
                        if (contains_unescaped_newline(token))
                        {
                            recently_seen_pound_define = false;
                            recently_seen_paren_0 = false;

                            // this is an end of a potential declaration
                            token_accum.swap(last_full_declaration);
                            token_accum.clear();
                        }
                    }
                }break;

            default: // ------------------------------------------
                {
                    token_accum.push_back(make_pair(type,token));
                    at_top_of_new_scope = false;
                }break;
        }


        tok.get_token(type, token);
    }
}

// ----------------------------------------------------------------------------------------

string get_function_name (
    const std::vector<std::pair<int,string> >& declaration
)
{
    string name;

    bool last_was_operator = false;
    bool seen_operator = false;
    for (unsigned long i = 0; i < declaration.size(); ++i)
    {
        if (declaration[i].first == tok_type::OTHER &&
            declaration[i].second == "(" && !last_was_operator )
        {
            if (i != 0 && !seen_operator)
            {
                // if this is a destructor then include the ~
                if (i > 1 && declaration[i-2].second == "~")
                    name = "~" + declaration[i-1].second;
                else
                    name = declaration[i-1].second;
            }

            break;
        }

        if (declaration[i].first == tok_type::KEYWORD &&
            declaration[i].second == "operator")
        {
            last_was_operator = true;
            seen_operator = true;
        }
        else
        {
            last_was_operator = false;
        }

        if (seen_operator)
        {
            if (name.size() != 0 && 
                (declaration[i].first == tok_type::IDENTIFIER || declaration[i].first == tok_type::KEYWORD) )
            {
                name += " ";
            }

            name += declaration[i].second;
        }

    }

    return name;
}

// ----------------------------------------------------------------------------------------

string pretty_print_declaration (
    const std::vector<std::pair<int,string> >& decl
)
{
    string temp;
    long angle_count = 0;
    long paren_count = 0;

    if (decl.size() == 0)
        return temp;

    temp = decl[0].second;


    bool just_closed_template = false;
    bool in_template = false;
    bool last_was_scope_res = false;
    bool seen_operator = false;

    if (temp == "operator")
        seen_operator = true;

    for (unsigned long i = 1; i < decl.size(); ++i)
    {
        bool last_was_less_than = false;
        if (decl[i-1].first == tok_type::OTHER && decl[i-1].second == "<")
            last_was_less_than = true;


        if (decl[i].first == tok_type::OTHER && decl[i].second == "<")
            ++angle_count;

        if (decl[i-1].first == tok_type::KEYWORD && decl[i-1].second == "template" && 
            decl[i].first == tok_type::OTHER && decl[i].second == "<")
        {
            in_template = true;
            temp += " <\n    ";
        }
        else if (decl[i].first == tok_type::OTHER && decl[i].second == ">")
        {
            --angle_count;
            if (angle_count == 0 && in_template)
            {
                temp += "\n>\n";
                just_closed_template = true;
                in_template = false;
            }
            else
            {
                temp += ">";
            }
        }
        else if (decl[i].first == tok_type::OTHER && decl[i].second == "<")
        {
            temp += "<";
        }
        else if (decl[i].first == tok_type::OTHER && decl[i].second == ",")
        {
            if (in_template || (paren_count != 0 && angle_count == 0))
                temp += ",\n   ";
            else
                temp += ",";
        }
        else if (decl[i].first == tok_type::OTHER && decl[i].second == "&")
        {
            temp += "&";
        }
        else if (decl[i].first == tok_type::OTHER && decl[i].second == "*")
        {
            temp += "*";
        }
        else if (decl[i].first == tok_type::KEYWORD && decl[i].second == "operator")
        {
            temp += "\noperator";
            seen_operator = true;
        }
        else if (decl[i].first == tok_type::OTHER && decl[i].second == ":" &&
                 (decl[i-1].second == ":" || (i+1<decl.size() && decl[i+1].second == ":") ) )
        {
            temp += ":";
            last_was_scope_res = true;
        }
        else if (decl[i].first == tok_type::OTHER && decl[i].second == "(")
        {
            const bool next_is_paren = (i+1 < decl.size() && decl[i+1].first == tok_type::OTHER && decl[i+1].second == ")");

            if (paren_count == 0 && next_is_paren == false)
                temp += " (\n   ";
            else
                temp += "(";

            ++paren_count;
        }
        else if (decl[i].first == tok_type::OTHER && decl[i].second == ")")
        {
            --paren_count;
            if (paren_count == 0 && decl[i-1].second != "(")
                temp += "\n)";
            else
                temp += ")";
        }
        else if (decl[i].first == tok_type::IDENTIFIER && i+1 < decl.size() &&
                 decl[i+1].first == tok_type::OTHER && decl[i+1].second == "(")
        {
            if (just_closed_template || paren_count != 0 || decl[i-1].second == "~")
                temp += decl[i].second;
            else if (seen_operator)
                temp += " " + decl[i].second;
            else
                temp += "\n" + decl[i].second;

            just_closed_template = false;
            last_was_scope_res = false;
        }
        else
        {
            if (just_closed_template || last_was_scope_res || last_was_less_than || 
                (seen_operator && paren_count == 0 && decl[i].first == tok_type::OTHER ))
                temp += decl[i].second;
            else
                temp += " " + decl[i].second;

            just_closed_template = false;
            last_was_scope_res = false;
        }



    }

    return temp;
}

// ----------------------------------------------------------------------------------------

string format_comment (
    const string& comment
)
{
    if (comment.size() <= 6)
        return "";

    string temp = trim(trim(comment.substr(3,comment.size()-6), " \t"), "\n\r");

    // now figure out what the smallest amount of leading white space is and remove it from each line.
    unsigned long num_whitespace = 100000;
    
    string::size_type pos1 = 0, pos2 = 0;

    while (pos1 != string::npos)
    {
        // find start of non-white-space
        pos2 = temp.find_first_not_of(" \t",pos1);

        // if this is a line of just white space then ignore it
        if (pos2 != string::npos && temp[pos2] != '\n' && temp[pos2] != '\r')
        {
            if (pos2-pos1 < num_whitespace)
                num_whitespace = pos2-pos1;
        }

        // find end-of-line
        pos1 = temp.find_first_of("\n\r", pos2);
        // find start of next line
        pos2 = temp.find_first_not_of("\n\r", pos1);
        pos1 = pos2;
    }

    // now remove the leading white space
    string temp2;
    bool seen_newline = true;
    unsigned long counter = 0;
    for (unsigned long i = 0; i < temp.size(); ++i)
    {
        // if we are looking at a new line
        if (temp[i] == '\n' || temp[i] == '\r')
        {
            counter = 0;
        }
        else if (counter < num_whitespace)
        {
            ++counter;
            continue;
        }

        temp2 += temp[i];
    }

    return temp2;
}

// ----------------------------------------------------------------------------------------

typedef_record convert_tok_typedef_record (
    const tok_typedef_record& rec
)
{
    typedef_record temp;
    temp.declaration = pretty_print_declaration(rec.declaration);
    return temp;
}

// ----------------------------------------------------------------------------------------

variable_record convert_tok_variable_record (
    const tok_variable_record& rec
)
{
    variable_record temp;
    temp.declaration = pretty_print_declaration(rec.declaration);
    return temp;
}

// ----------------------------------------------------------------------------------------

method_record convert_tok_method_record (
    const tok_method_record& rec
)
{
    method_record temp;

    temp.comment = format_comment(rec.comment);
    temp.name = get_function_name(rec.declaration);
    temp.declaration = pretty_print_declaration(rec.declaration);
    return temp;
}

// ----------------------------------------------------------------------------------------

class_record convert_tok_class_record (
    const tok_class_record& rec
)
{
    class_record crec;


    crec.scope = rec.scope;
    crec.file = rec.file;
    crec.comment = format_comment(rec.comment);

    crec.name.clear();

    // find the first class token
    for (unsigned long i = 0; i+1 < rec.declaration.size(); ++i)
    {
        if (rec.declaration[i].first == tok_type::KEYWORD &&
            (rec.declaration[i].second == "class" ||
            rec.declaration[i].second == "struct" )
            )
        {
            crec.name = rec.declaration[i+1].second;
            break;
        }
    }

    crec.declaration = pretty_print_declaration(rec.declaration);

    for (unsigned long i = 0; i < rec.public_typedefs.size(); ++i)
        crec.public_typedefs.push_back(convert_tok_typedef_record(rec.public_typedefs[i]));

    for (unsigned long i = 0; i < rec.public_variables.size(); ++i)
        crec.public_variables.push_back(convert_tok_variable_record(rec.public_variables[i]));

    for (unsigned long i = 0; i < rec.public_methods.size(); ++i)
        crec.public_methods.push_back(convert_tok_method_record(rec.public_methods[i]));

    for (unsigned long i = 0; i < rec.public_inner_classes.size(); ++i)
        crec.public_inner_classes.push_back(convert_tok_class_record(rec.public_inner_classes[i]));


    return crec;
}

// ----------------------------------------------------------------------------------------

function_record convert_tok_function_record (
    const tok_function_record& rec
)
{
    function_record temp;

    temp.scope = rec.scope;
    temp.file = rec.file;
    temp.comment = format_comment(rec.comment);
    temp.name = get_function_name(rec.declaration);
    temp.declaration = pretty_print_declaration(rec.declaration);

    return temp;
}

// ----------------------------------------------------------------------------------------

void convert_to_normal_records (
    const std::vector<tok_function_record>& tok_functions,
    const std::vector<tok_class_record>& tok_classes,
    std::vector<function_record>& functions,
    std::vector<class_record>& classes
)
{
    functions.clear();
    classes.clear();


    for (unsigned long i = 0; i < tok_functions.size(); ++i)
    {
        functions.push_back(convert_tok_function_record(tok_functions[i]));
    }


    for (unsigned long i = 0; i < tok_classes.size(); ++i)
    {
        classes.push_back(convert_tok_class_record(tok_classes[i]));
    }


}

// ----------------------------------------------------------------------------------------

string add_entity_ref (const string& str)
{
    string temp;
    for (unsigned long i = 0; i < str.size(); ++i)
    {
        if (str[i] == '&')
            temp += "&amp;";
        else if (str[i] == '<')
            temp += "&lt;";
        else if (str[i] == '>')
            temp += "&gt;";
        else
            temp += str[i];
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

void write_as_xml (
    const function_record& rec,
    ostream& fout
)
{
    fout << "    <function>\n";
    fout << "      <name>"        << add_entity_ref(rec.name)        << "</name>\n";
    fout << "      <scope>"       << add_entity_ref(rec.scope)       << "</scope>\n";
    fout << "      <declaration>" << add_entity_ref(rec.declaration) << "</declaration>\n";
    fout << "      <file>"        << add_entity_ref(rec.file)        << "</file>\n";
    fout << "      <comment>"     << add_entity_ref(rec.comment)     << "</comment>\n";
    fout << "    </function>\n";
}

// ----------------------------------------------------------------------------------------

void write_as_xml (
    const class_record& rec,
    ostream& fout,
    unsigned long indent 
)
{
    const string pad(indent, ' ');

    fout << pad << "<class>\n";
    fout << pad << "  <name>"        << add_entity_ref(rec.name)        << "</name>\n";
    fout << pad << "  <scope>"       << add_entity_ref(rec.scope)       << "</scope>\n";
    fout << pad << "  <declaration>" << add_entity_ref(rec.declaration) << "</declaration>\n";
    fout << pad << "  <file>"        << add_entity_ref(rec.file)        << "</file>\n";
    fout << pad << "  <comment>"     << add_entity_ref(rec.comment)     << "</comment>\n";


    if (rec.public_typedefs.size() > 0)
    {
        fout << pad << "  <public_typedefs>\n";
        for (unsigned long i = 0; i < rec.public_typedefs.size(); ++i)
        {
            fout << pad << "    <typedef>"        << add_entity_ref(rec.public_typedefs[i].declaration)        << "</typedef>\n";
        }
        fout << pad << "  </public_typedefs>\n";
    }


    if (rec.public_variables.size() > 0)
    {
        fout << pad << "  <public_variables>\n";
        for (unsigned long i = 0; i < rec.public_variables.size(); ++i)
        {
            fout << pad << "    <variable>"        << add_entity_ref(rec.public_variables[i].declaration)        << "</variable>\n";
        }
        fout << pad << "  </public_variables>\n";
    }


    if (rec.public_methods.size() > 0)
    {
        fout << pad << "  <public_methods>\n";
        for (unsigned long i = 0; i < rec.public_methods.size(); ++i)
        {
            fout << pad << "    <method>\n";
            fout << pad << "      <name>"        << add_entity_ref(rec.public_methods[i].name)        << "</name>\n";
            fout << pad << "      <declaration>" << add_entity_ref(rec.public_methods[i].declaration) << "</declaration>\n";
            fout << pad << "      <comment>"     << add_entity_ref(rec.public_methods[i].comment)     << "</comment>\n";
            fout << pad << "    </method>\n";
        }
        fout << pad << "  </public_methods>\n";
    }


    if (rec.public_inner_classes.size() > 0)
    {
        fout << pad << "  <public_inner_classes>\n";
        for (unsigned long i = 0; i < rec.public_inner_classes.size(); ++i)
        {
            write_as_xml(rec.public_inner_classes[i], fout, indent+4);
        }
        fout << pad << "  </public_inner_classes>\n";
    }


    fout << pad << "</class>\n";
}

// ----------------------------------------------------------------------------------------

void save_to_xml_file (
    const std::vector<function_record>& functions,
    const std::vector<class_record>& classes
)
{
    ofstream fout("output.xml");

    fout << "<code>" << endl;

    fout << "  <classes>" << endl;
    for (unsigned long i = 0; i < classes.size(); ++i)
    {
        write_as_xml(classes[i], fout, 4);
        fout << "\n";
    }
    fout << "  </classes>\n\n" << endl;


    fout << "  <global_functions>" << endl;
    for (unsigned long i = 0; i < functions.size(); ++i)
    {
        write_as_xml(functions[i], fout);
        fout << "\n";
    }
    fout << "  </global_functions>" << endl;

    fout << "</code>" << endl;
}

// ----------------------------------------------------------------------------------------

void generate_xml_markup(
    const cmd_line_parser<char>::check_1a_c& parser, 
    const std::string& filter, 
    const unsigned long search_depth
)
{

    // first figure out which files should be processed
    std::vector<std::pair<string,string> > files; 
    obtain_list_of_files(parser, filter, search_depth, files);

    cout << "Generating output.xml..." << endl;
    cout << "Number of files: " << files.size() << endl;

    std::vector<tok_function_record> tok_functions;
    std::vector<tok_class_record> tok_classes;

    for (unsigned long i = 0; i < files.size(); ++i)
    {
        ifstream fin(files[i].second.c_str());
        if (!fin)
        {
            cerr << "Error opening file: " << files[i].second << endl;
            return;
        }
        process_file(fin, files[i].first, tok_functions, tok_classes); 
    }

    cout << "\nNumber of functions found: " << tok_functions.size() << endl;
    cout << "Number of classes found:   " << tok_classes.size() << endl;
    cout << endl;

    //cout << tok_functions[0].comment << endl;

    std::vector<function_record> functions;
    std::vector<class_record> classes;

    convert_to_normal_records(tok_functions, tok_classes, functions, classes);

    save_to_xml_file(functions, classes);
}

// ----------------------------------------------------------------------------------------

