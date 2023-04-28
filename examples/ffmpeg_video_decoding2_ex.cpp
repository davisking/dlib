// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers, 
    in this case the decoding API.

    This is a pretty simple example. It loads a raw codec file, parses chunks of 
    data to the decoder, plots images to a GUI window, and counts audio samples.
*/

#include <cstdio>
#include <dlib/media.h>
#include <dlib/gui_widgets.h>
#include <dlib/cmd_line_parser.h>

using namespace std;
using namespace dlib;
using namespace dlib::ffmpeg;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("i",      "input video encoded stream. e.g. dlib/test/ffmpeg_data/MOT20-08-raw.h264", 1);
    parser.add_option("codec",  "codec name. e.g. h264", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    const std::string filepath = parser.option("i").argument();
    const std::string codec    = parser.option("codec").argument();

    decoder dec([&] {
        decoder::args args;
        args.codec_name = codec;
        return args;
    }());

    if (!dec.is_open())
    {
        printf("Failed to create decoder.\n");
        return EXIT_FAILURE;
    }

    frame                   f;
    array2d<rgb_pixel>      img;
    ffmpeg::decoder_status  status{ffmpeg::DECODER_EAGAIN};
    int                     samples{0};

    // When reading frames, we get exactly what's in the codec by default.
    // To resize, change pixel format, resample or change sample format, 
    // you have to pass extra arguments to read() which either resize or resample
    // the frame. Since we want rgb_pixel, we need to set the pixel format appropriately.
    const resizing_args args_image {0, 0, pix_traits<rgb_pixel>::fmt};

    image_window win;

    const auto pull = [&]
    {
        while ((status = dec.read(f, args_image)) == ffmpeg::DECODER_FRAME_AVAILABLE)
        {
            if (f.is_image())
            {
                convert(f, img);
                win.set_image(img);
            }
            else if (f.is_audio())
            {
                samples += f.nsamples();
            }
        }
    };

    ifstream fin{filepath, std::ios::binary};
    std::vector<char> buf(1024);

    while (fin && status != ffmpeg::DECODER_CLOSED)
    {
        fin.read(buf.data(), buf.size());
        size_t ret = fin.gcount();
        dec.push_encoded((const uint8_t*)buf.data(), ret);
        pull();
    }

    dec.flush();
    pull();

    printf("Read %i audio samples\n", samples);

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}