// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers, in this case the encding API.

    This is a pretty simple example. It loads a video file, extracts the images and
    re-encodes them into a raw buffer using a user-specified codec.
    
    Please see the following examples on how to decode, demux, and get information on your installation of ffmpeg:
        - ffmpeg_info_ex.cpp
        - ffmpeg_video_decoding_ex.cpp
        - ffmpeg_video_demuxing_ex.cpp
*/

#include <cstdio>
#include <dlib/media.h>
#include <dlib/cmd_line_parser.h>

using namespace std;
using namespace dlib;
using namespace dlib::ffmpeg;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("i",      "input video", 1);
    parser.add_option("codec",  "codec name. e.g. h264. Defaults to mpeg4", 1);
    parser.add_option("height", "height of encoded stream. Defaults to whatever is in the video file", 1);
    parser.add_option("width",  "width of encoded stream. Defaults to whatever is in the video file", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i", "codec", "height", "width"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    if (!parser.option("i"))
    {
        cout << "Missing -i" << endl;
        parser.print_options();
        return 0;
    }

    const std::string filepath = get_option(parser, "i", "");
    const std::string codec    = get_option(parser, "codec", "mpeg4");

    // Check your codec is available
    const auto codecs = ffmpeg::list_codecs();

    const bool codec_available = std::find_if(begin(codecs),   
                                              end(codecs),   
                                              [&](const auto& c) {return c.codec_name == codec && c.supports_encoding;}) != codecs.end();

    if (!codec_available)
    {
        cout << "Codec `" << codec << "` is not available as an encoder in your installation of ffmpeg." << endl;
        cout << "Either choose another codec, or build ffmpeg from source with the right dependencies installed." << endl;
        cout << "For example, if you are trying to encode to h264, hevc/h265, vp9 or avi, then your installation of ffmpeg ";
        cout << "needs to link to libx264, libx265, libvp9 or libav1" << endl;
        return EXIT_FAILURE;
    }

    // Encode to multiple different types of buffers.
    const auto encode = [&](auto& out)
    {
        demuxer cap({filepath, video_enabled, audio_disabled});

        if (!cap.is_open() || !cap.video_enabled())
        {
            cout << "Failed to open " << filepath << endl;
            return;
        }

        encoder enc([&] {
            encoder::args args;
            args.args_codec.codec_name  = codec;
            args.args_image.h           = get_option(parser, "height", cap.height());
            args.args_image.w           = get_option(parser, "width",  cap.width());
            args.args_image.framerate   = cap.fps();
            return args;
        }(), sink(out));

        frame f;
        while (cap.read(f))
            enc.push(std::move(f));
        
        // enc.flush(); 
        // You don't have to call flush() here because it's called in the destructor of encoder
        // If you call it more than once, it becomes a no-op basically.
    };

    std::vector<char>       buf1;
    std::vector<int8_t>     buf2;
    std::vector<uint8_t>    buf3;
    std::ostringstream      buf4;
    std::ofstream           buf5("encoded.h264", std::ios::binary);

    encode(buf1);
    encode(buf2);
    encode(buf3);
    encode(buf4);
    encode(buf5);

    cout << "vector<char>       size " << buf1.size() << endl;
    cout << "vector<int8_t>     size " << buf2.size() << endl;
    cout << "vector<uint8_t>    size " << buf3.size() << endl;
    cout << "ostringstream      size " << buf4.tellp() << endl;
    cout << "ofstream           size " << buf5.tellp() << endl;

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}