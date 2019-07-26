// Copyright (C) 2019  Paul Dreik (github@pauldreik.se)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/xml_parser.h>
#include <sstream>

namespace {

class doc_handler : public dlib::document_handler
{
public:

    virtual void start_document ()
    {
    }

    virtual void end_document (
    )
    {
    }

    virtual void start_element (
        const unsigned long line_number,
        const std::string& name,
        const dlib::attribute_list& atts
    )
    {
        // print all the tag's attributes
        atts.reset();
        while (atts.move_next())
        {
            // do something with atts, to access the data.
            //cout << "\tattribute: " << atts.element().key() << " = " << atts.element().value() << endl;
        }
    }

    virtual void end_element (
        const unsigned long line_number,
        const std::string& name
    )
    {
    }

    virtual void characters (
        const std::string& data
    )
    {
    }

    virtual void processing_instruction (
        const unsigned long line_number,
        const std::string& target,
        const std::string& data
    )
    {
    }

}; // class
} // anon. namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, std::size_t Size) {

    // prevent large input, just to keep the fuzzing speed up.
    if(Size > 200) {
        return 0;
    }

    std::stringstream iss;
    iss.write((const char*)Data,Size);

    try {
        dlib::xml_parser parser;
        doc_handler dh;
        parser.add_document_handler(dh);
        parser.parse(iss);
    } catch(...) {
    }

    return 0;
}
