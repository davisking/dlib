// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating some of the functional ffmpeg wrapper APIs.
    It demonstrates how to list the supported codecs, muxers, demuxers and protocols
    in your installation of ffmpeg (or the one dlib built against).

    It also demonstrates how to check ffmpeg library versions programmatically.
*/

#include <cstdio>
#include <dlib/media/ffmpeg_utils.h>

using namespace std;
using namespace dlib;

int main()
{
    // List all codecs supported by this installation of ffmpeg libraries
    const auto codecs = ffmpeg::list_codecs();
    printf("Supported codecs:\n");
    for (const auto& codec : codecs)
        printf("name : %-16s : encoding supported %i decoding supported %i\n", codec.codec_name.c_str(), codec.supports_encoding, codec.supports_decoding);

    printf("\n");

    // List all demuxers supported by this installation of ffmpeg libraries
    const auto demuxers = ffmpeg::list_demuxers();
    printf("Supported demuxers:\n");
    for (const auto& demuxer : demuxers)
        printf("%s\n", demuxer.c_str());

    printf("\n");
    
    // List all muxers supported by this installation of ffmpeg libraries
    const auto muxers = ffmpeg::list_muxers();
    printf("Supported muxers:\n");
    for (const auto& muxer : muxers)
        printf("%s\n", muxer.c_str());

    printf("\n");

    printf("Can read MP4 file with H264 encoded video stream?\n");

    const bool mp4_available = 
        std::find_if(begin(demuxers),   
                     end(demuxers), 
                     [](const auto& demux) {return demux.find("mp4") != std::string::npos;}) != demuxers.end();

    const bool h264_available = 
        std::find_if(begin(codecs),   
                     end(codecs),   
                     [](const auto& codec) {return codec.codec_name == "h264" && codec.supports_decoding;}) != codecs.end();

    printf("Anwser: %i\n", mp4_available && h264_available);

    return EXIT_SUCCESS;
}