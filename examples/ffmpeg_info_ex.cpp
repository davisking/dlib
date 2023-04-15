// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating some of the functional ffmpeg wrapper APIs.
    It demonstrates how to list the supported codecs, muxers, demuxers and protocols
    in your installation of ffmpeg (or the one dlib built against).

    It also demonstrates how to check ffmpeg library versions programmatically.
*/

#include <iostream>
#include <iomanip>
#include <dlib/media/ffmpeg_utils.h>

using namespace std;
using namespace dlib;

int main()
{
    // List all codecs supported by this installation of ffmpeg libraries
    const auto codecs = ffmpeg::list_codecs();
    
    cout << "Supported codecs:\n";
    for (const auto& codec : codecs)
        cout << "    name : " << left << setw(20) << codec.codec_name << " id : " << codec.codec_id << " : encoding supported " << codec.supports_encoding << " decoding supported " << codec.supports_decoding << '\n';
    cout << '\n';

    // List all demuxers supported by this installation of ffmpeg libraries
    const auto demuxers = ffmpeg::list_demuxers();

    cout << "Supported demuxers:\n";
    for (const auto& demuxer : demuxers)
        cout << "    name : " << demuxer << '\n';
    cout << '\n';
    
    // List all muxers supported by this installation of ffmpeg libraries
    cout << "Supported muxers:\n";
    for (const auto& muxer : ffmpeg::list_muxers()) 
    {
        cout << "    name : " << muxer.name << '\n';
        if (!muxer.supported_codecs.empty())
        {
            cout << "        supported codecs:\n";
            for (const auto& codec : muxer.supported_codecs)
                cout << "            " << codec.codec_name << '\n';
        }
    }
    cout << '\n';

    // List all input devices supported by this installation of ffmpeg libraries
    cout << "Supported input devices:\n";
    for (const auto& device :  ffmpeg::list_input_device_types())
    {
        cout << "    device type : `" << device.device_type << "` is audio " << device.is_audio_type << " is video " << device.is_video_type << '\n';

        const auto instances = ffmpeg::list_input_device_instances(device.device_type);
        if (!instances.empty())
        {
            cout << "        instances :\n";
            for (const auto& instance : instances)
                cout << "            name : " << left << setw(32) << instance.name << ", description : " << instance.description << '\n';
        }
    }

    cout << '\n';

    // List all input devices supported by this installation of ffmpeg libraries
    cout << "Supported output devices:\n";
    for (const auto& device :  ffmpeg::list_output_device_types())
    {
        cout << "    device type : `" << device.device_type << "` is audio " << device.is_audio_type << " is video " << device.is_video_type << '\n';

        const auto instances = ffmpeg::list_output_device_instances(device.device_type);
        if (!instances.empty())
        {
            cout << "        instances :\n";
            for (const auto& instance : instances)
                cout << "            name : " << left << setw(32) << instance.name << ", description : " << instance.description << '\n';
        }
    }

    cout << '\n';
    
    const bool mp4_available = 
        std::find_if(begin(demuxers),   
                     end(demuxers), 
                     [](const auto& demux) {return demux.find("mp4") != std::string::npos;}) != demuxers.end();

    const bool h264_available = 
        std::find_if(begin(codecs),   
                     end(codecs),   
                     [](const auto& codec) {return codec.codec_name == "h264" && codec.supports_decoding;}) != codecs.end();

    cout << "Can read MP4 file with H264 encoded video stream? " << (mp4_available && h264_available) << '\n';

    return EXIT_SUCCESS;
}
