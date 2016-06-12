// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_UTILITIES_H_
#define DLIB_DNn_UTILITIES_H_

#include "core.h"
#include "utilities_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        class visitor_net_to_xml
        {
        public:

            visitor_net_to_xml(std::ostream& out_) : out(out_) {}

            template<typename input_layer_type>
            void operator()(size_t idx, const input_layer_type& l) 
            {
                out << "<layer idx='"<<idx<<"' type='input'>\n";
                to_xml(l,out);
                out << "</layer>\n";
            }

            template <typename T, typename U>
            void operator()(size_t idx, const add_loss_layer<T,U>& l) 
            {
                out << "<layer idx='"<<idx<<"' type='loss'>\n";
                to_xml(l.loss_details(),out);
                out << "</layer>\n";
            }

            template <typename T, typename U, typename E>
            void operator()(size_t idx, const add_layer<T,U,E>& l) 
            {
                out << "<layer idx='"<<idx<<"' type='comp'>\n";
                to_xml(l.layer_details(),out);
                out << "</layer>\n";
            }

            template <unsigned long ID, typename U, typename E>
            void operator()(size_t idx, const add_tag_layer<ID,U,E>& l) 
            {
                out << "<layer idx='"<<idx<<"' type='tag' id='"<<ID<<"'/>\n";
            }

            template <template<typename> class T, typename U>
            void operator()(size_t idx, const add_skip_layer<T,U>& l) 
            {
                out << "<layer idx='"<<idx<<"' type='skip' id='"<<(tag_id<T>::id)<<"'/>\n";
            }

        private:

            std::ostream& out;
        };
    }

    template <typename net_type>
    void net_to_xml (
        const net_type& net,
        std::ostream& out
    )
    {
        out << "<net>\n";
        visit_layers(net, impl::visitor_net_to_xml(out));
        out << "</net>\n";
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_H_ 



