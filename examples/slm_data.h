#ifndef SLM_DATA_H
#define SLM_DATA_H

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <dlib/compress_stream.h>
#include <dlib/base64.h>

// Dataset identification
/*!
    Dataset formats
        Each dataset, when decompressed, contains text with specific structure:

        - RAW_TEXT: Plain text without special separators
          Example: "First paragraph. Second paragraph. Third paragraph."

        - DELIMITED_TEXT: Text segments separated by "@@" delimiter
          Example: "Segment 1@@Segment 2@@Segment 3"

        - PAIRED_TEXT: Alternating text segments separated by "@@", grouped into pairs
          Example: "Question 1@@Answer 1@@Question 2@@Answer 2"
!*/
enum class dataset_id
{
    SHAKESPEARE_EXTRACT,    // Classic literature excerpt (RAW_TEXT format)
    SHAKESPEARE_PROMPT,     // Shakespeare text formatted as training prompt (RAW_TEXT format)
    INTERNAL_TRAINING,      // Internal training corpus (RAW_TEXT format)
    PHYSICS_PARAGRAPHS,     // Physics text segments (DELIMITED_TEXT format)
    BLACK_HOLE_QA           // Question-answer pairs on black holes (PAIRED_TEXT format)
};

// Code compression utility
namespace detail
{
    // Decompresses base64-encoded and compressed data
    // This is the low-level utility used by all dataset accessors
    inline std::string decompress_data(const std::string& compressed_base64_data)
    {
        dlib::base64 base64_coder;
        dlib::compress_stream::kernel_1ea compressor;
        std::ostringstream sout;
        std::istringstream sin;

        sin.str(compressed_base64_data);
        base64_coder.decode(sin, sout);

        sin.clear();
        sin.str(sout.str());
        sout.str("");

        compressor.decompress(sin, sout);
        return sout.str();
    }

    // Splits a string by the "@@" delimiter into a vector of segments
    // Used for DELIMITED_TEXT and PAIRED_TEXT formats
    inline std::vector<std::string> split_by_delimiter(const std::string& text)
    {
        std::vector<std::string> result;
        std::string::size_type start = 0;
        std::string::size_type end;
        const std::string delimiter = "@@";

        while ((end = text.find(delimiter, start)) != std::string::npos)
        {
            result.push_back(text.substr(start, end - start));
            start = end + delimiter.length();
        }

        // Add last segment if not empty
        if (start < text.length())
            result.push_back(text.substr(start));

        return result;
    }

    // Converts a delimited string into pairs by grouping consecutive segments
    // Expects even number of segments for proper pairing
    inline std::vector<std::pair<std::string, std::string>> parse_pairs(const std::string& text)
    {
        auto segments = split_by_delimiter(text);

        if (segments.size() % 2 != 0)
            throw std::runtime_error("Paired dataset must have even number of segments");

        std::vector<std::pair<std::string, std::string>> result;
        result.reserve(segments.size() / 2);

        for (size_t i = 0; i < segments.size(); i += 2)
            result.emplace_back(segments[i], segments[i + 1]);

        return result;
    }
}

// Compressed dataset storage
namespace datasets
{
    // Returns compressed Shakespeare extract data.
    // Decompressed format : RAW_TEXT(plain continuous text)
    inline std::string get_shakespeare_compressed()
    {
        std::ostringstream sout;
        sout << "UU0b45RDWgwOJdW8F9arj/rWvyYXKU0ZCZOrQSPZnFRPBpS4tJ4b3jPEo37TnIKnJUijuYMKL5Xh";
        sout << "voFfev7RxQRDb0u6xe1EtpbzTtSiLfocXEdb8ajZq6LneT8w0p4FxnOR1+ulQWVJAA/UbHnl9Y98";
        sout << "Uk7Ds5sfmRdsCjDcAUSvnyBbJa54JtzPBVGBkK0i2/BUroFQOr4dq255RYLc3uVPl45dyfVhW7KR";
        sout << "JxgKvj/Vg4bqBXk4YjRU9XCE/Jhh9uwUJEyZLse8UcoPvo1qp+yOc+sI+zlf3Iat1y+DyM59rxXT";
        sout << "mLm1V7uLNHSq9c/z0DkiuXFM0L4catKL27o77AB3V75LhDazEafyuRbf/vXa9BEUqzDQXvAsfT/m";
        sout << "AT2zBS/h3+K3WN1xI0JuClFAwK+epbnciH0Xown9qFKRktdbngnzggPCKDAs+PaCRiDL7V6XMvYk";
        sout << "TcdJ2Ecce28uCcUSyg1IOuAFSfzYpfDxpkFbKSuDI6jh2/z51LUX2gt/cUNIwORr1v9uBiqlLs1O";
        sout << "kWO3MQbrtTLbMXKevNrPrADcAm/U+TSziKu9if+OCY/g/rJc4KrS+d1oKZARs4xU3st/HKEcC+L2";
        sout << "QdHtstPTtUZDzWSqQwVouDeueNEFQC/8oPCwlcNQMY4bRuf1D7rNZTuoJVJ3YuRRtIZW9uGYeP9j";
        sout << "PbC9kTd6+4ROF84JZ6/1w914n+2hLJxuVy876ohwwJ6bcpOexdmaQe9+oy+MTNTfaPkr4DOa/wie";
        sout << "W1u2VkssLkWS3NpND0UO5ZivEvOjFTEYzDrY6/PQrYA4QbTkMBQjykVEXIEV/8m3h022Yza2ugr0";
        sout << "osnyggIknb5n7BDZ9JKDPa4EJXEkrDlmdM/cZ8Gc2oqFf0Iu7iBfLc3xKEngK8JHsDCTlZLBhqxy";
        sout << "E1kKWxHr1UJu9lTWpGQ9R8McfInXXf9hQp42LFnFYimbMCzTE3zMBKs2PAfUr3euLfK3nSXQCU1F";
        sout << "lBVSPHAUMZdpeD3arP5XDV7XeX44gvyOg/js+jj05Gna2tqTIYg0zIi14Ovw2Oe2EJMYfVq8/Mro";
        sout << "KKCmgl/1KJD40fNwRFZTryoVVWqBjBuTFgwSsYjvSxf/Um7/ZXKe8cwyW+gP7C71tKlej6mmEktC";
        sout << "UO0kt49nBsqGmW/jMz7I255DhkU8zqUDqpOmPMoi1g1CRyDGmoSEKtq0YN0UYE7rR8cBK9AD0XYH";
        sout << "iltse5Hx2gSXcmVYQPemOmsjhjVEbeKQ1p9XspaTI0m3oM5/3KlyoM9bc5bIFbzdcL9wHATLEK+P";
        sout << "LShITqooveRLPSQ73gwmAtzy2btiqm9I9xCT/TBYs7g7RJON5lfX+w2Uw7Xpfqys64+4r1QvNks6";
        sout << "TWfJFpVjNpLDvq2u+FTR+dCjqi7x2nWl9YRhfiapBW0bG2aqa1sdVh9/4dp97zMueqsWCGBUJaW5";
        sout << "IdnsZMFWE6sfII5+HnFQOtcgyFRlelAW0xPiRh/T30BoiqgcEKWgAZ+CpAq4I7nxkiq2ZiojTJPm";
        sout << "skvRgdimillgrrUuhe5G6FNnDceclsf8VB9nKMiLklAVoNlyo/pwvjwisQpTlGtS3fCxUzEHlsk5";
        sout << "FPaLx/OO34c//Py53TXw931JSipD7b/41hSkAFtAOsCuU/hYZUK2b/tfy4bzftc/YPHKTRzO6dsH";
        sout << "U3S3j98I2PEnK8sUsuUKx0Xk2KECZ0KEOFlcGw+S9/qmHdObkN2HK8NyB6vvlsBxUwqKbn3xbszS";
        sout << "ysuAE2LPBzve3JYw/NHhjUWYCzZYksJITGHTotV/Etv780b1pbQyMTm8kagG17o6Bk1CWVIpuVhy";
        sout << "3be8RMYzlvDP3a4VmSGOJBapzH6KCyCzNszDB4Zb4teSekSHGPxXdpJK+FMJOmL066lZBOvaIoSN";
        sout << "7RdfEfUh+EIzlF0gnHUolZpnaI+lX9VgfZ6X3TgtLY/EBSP6iGfXZqi/PfgGHSKRx0MtSeVWuUUM";
        sout << "AkN4ARbo90ke74bsx32zZYG5waW/rkkHE8e3fQ75tTUgXhf47M5SZLWrxl/qDEbJL3b+H2dxkD+c";
        sout << "+GbDbF8nzyn3eZPlyuxgo6vJpE3APjPRqrqv26HN4GepnRPK2kr6o+fhdrItBYAgakrL7NzWhH6P";
        sout << "itdDrlx61WKhuv1YUc1KkGeaxfx+DuIsy/mpwwUn1D5xpGtkTeruXsfMJvhDzuNCYW60k0ryEHJS";
        sout << "E8vUh9vwMaa9Mc38+4aHtHvJiahy1zHLUspDFcYIC5ubpmmT8fJVBbo0H3qy1j++p65v0ktL/Szm";
        sout << "68w0+KejbIrLuL5HG/5VpCNId1DpavclTUJKsEgoUHZILtA+OM12SW2b5MgIofntxqWpqbfGErNh";
        sout << "ICwgDSJfyLiPHvRtS9nKxveHWnT0yzEnUdEZY4xiHtDmW/Qt/LsBzIDplVqTh1vofdBkeGC/LMPP";
        sout << "6F+Nq94rKkIXHZf26H+C53nRTPshYO+xMFcI421jyRLr0AKBLZkcnf4C7jfU5xy41d/iXA9YsM5m";
        sout << "A+IUUB/GIoCswjO1Sl4JxaxAW/fDqUNG2wGFHimeclb4Gqyiv/hPkujRR1QueVMtp1F7D/QP92gF";
        sout << "dT+h8WRmZZWiPvS/ngxGF/drgPt4swYBHzTjdz/Qfc+AepV+gnhkZ5p0H6n0kRy4QgaERvQbm2tc";
        sout << "J3v4epl+0eq7TMI7Xlw6EI1sap5iR2L+DyZfc7nn+YJQg8li0i9MbBxoxW0dG0GgBi6kCsGpPtnj";
        sout << "XCxUCNaMb19ODctd+vZ6zvGb5q7P8yqqdCQ1epi2imesfhfd9HfqgdgOE9gNOzIaY31ZuiHQCybE";
        sout << "cyRT8z3iNHW/GkvCuI7efitJ5Bc2/1saSbD/Vvskf70L3yB+yhruUdZGh5Vr2AO/olblT230W65m";
        sout << "K52GOEb0y6lPq9zKndk5CJ6qvFepqtzyej4Q3t7Zw55w+Fr1LptWxN6aknOqtQbFcStb2TbX9zMg";
        sout << "INPZoTR0UlkwRGBHZMYyVy4gOIRPaRFl5s3zLKuRFQwLp5nnZ5+blODezmwIbaaIRcY4FZGpBpmy";
        sout << "rdkL7GcxrCp8RoRj1ztX7M522USXugFK7PmuaIh7yXt+LunMi+3J0895Mg5GOMSGr7Fb5EdeGEcb";
        sout << "Na5lBbTzVqekYhHssz3O8q0EJcbqUvyjz+LsdLVastnD3zVh+4tHkSdZTZLK2DWxh4TgdoDYSDca";
        sout << "LAR18ViJ956Wu8062hXVoJMG3k3+MMzvCh/oCBj461qqD/haPMU1UyZt4ksbkVCc2I5p3EqQ6Fr2";
        sout << "lpArQWE9gBdfH+duCZ0dJLY9wuzxjfu44F5ip5BQ9qmqXRGejNOW+IkMfp1txBxcEVHNWW2CzoCG";
        sout << "E5KEPTv9yNxB3zwn752gH2TB9h7//kSp0npOL/RjF+5iiAnFWKJXm3RI6ZIyQPWQKlj9j0Wq9cne";
        sout << "NligkG7Go2/O6H07pAefCTARHMX054miZGKk3OnofVh3n+5kuvPrGmqDCoHyPqcBi6ah6p9mRiya";
        sout << "/+wEi6HlJElZijyej7MZ/+ocR6kyZx/+SUYa1HFrkr3cwnfZ6acoxcv3uCfq/UcRdoxjJsvNH+3k";
        sout << "xnK6zZZcl8LJ6+/+9gTM/+ogpUtVLkK9vZGEq+ip3FixZhbEeniHDbhQql895ksecx6dGCwOSyod";
        sout << "UaUkxq4Agr7Mkxi6tq2UE46kO4zOoLg0xdymgviXfvd1+Wplg1fngAG3OEVKMLy2qmm37426mp8T";
        sout << "cKGVZY2w2/k/7WiMbc7xGkZI1ToE2TCSSG4iB8U+2XIJdPUAahpJmkX7RzEmDn/2KF/qLWzUescp";
        sout << "sQLJmuTTp8A0DBpw6PVIt93eUM31TiUT9Kmwtkm4bc9id1y/AUSVtOQ+U0i+JoVFYPb6rsBOXfHp";
        sout << "y74t71l3JPNTu4ZI0P7yyYW5iq+WRosS56hJUBmK8zav7z84pit0wHWh04HWG9ZT2wLef0ANDWSG";
        sout << "z8oz7LpFHCmt3xdaCP4bsoZGaEMaEEJri1RjtKyXkun/EdrdSPVqJ1TWvN0UHk4ZFPL5ooHB7QyP";
        sout << "HXM+ImNLzh70hwifqL0/oj78ZprB2KUWm5gS2Y0OEcUMdZi3caqx7QS/5W6QjjDw+EY1bzTG7HOH";
        sout << "or+7qAWWuYWwwwijgs103rr/uwDhTtjzV0tnbR5ZROOsX+T7FyW283Tf2ZFq2aq1OWx9BNmfnZEt";
        sout << "1Zpg5m95cRP26GIgQQleMNfxslm2qsJPiaoJAXa/gygUkT8eDP/uFCOwtiY3oV36ej5uEoZyIn8B";
        sout << "g5vXtdLMihgc9AEoTfAvd23r02DXmkf1r/bnypKc1zNsLadROVjthB5GfzGrQOCVCXjQgUPAVR0a";
        sout << "XzBOKtND+cTglNtMVDYfk7SFFsToBAL2YblVgoLpgRUSZxANRYf4O6hIVJqPue8j5+qLAjSWXiSg";
        sout << "CzdR9AEq+Vas/vT6C/O1nNgMy+JKstKxN4BesKChvMNeIVTralP1TY0QWq5eyA/91JuK9/8PK11g";
        sout << "rzLVjYsq2BRmQmkEsUZEg61/poB+ZH+E9bHrjb5vwCFDGpRCpXC5GbQLhO/DHFCOHZXOCix+oP21";
        sout << "dkt1eHlgn2IS3FViXt/aWyrkhkmOc6jkJwzeC6wROT+2l7uWr4yeo1Ps2+naogR8fSHhYlvmCrPV";
        sout << "aSbUStBzmn5pQi3gGib9PVbCQ+Kft7VBUcJFmVH1Lle03By4Qd5wT51f4sNS1UL43nda0XLyWQ8H";
        sout << "kJY2n3pOk3C0dZjQdAx3rYyV26RMzwBpOC3uHnZa25kt+CWWGrbsapu5GBqkmfghPHREhYFB0zcy";
        sout << "9bZd6dVCbYeI7LwkhlGW/VvqnhaOnLzVmNkq0vvnBT8wt+EBIlvCLZDHcG3LAqZZoH56ZILuRs1f";
        sout << "RL3NwcPYI2rvbRyLc2ARJtfRjesxpfb0XDTddAWcjgsfytxlWxE3BjWVjHher9P2dLfrrkDgAdlW";
        sout << "i9MtjPSkHykxz9TzcBRATOphvwT5mwHWz9h/12vLM5iJxgfvW5LbGtkHxaCN2bM4/PiABAeGdFGc";
        sout << "SLkStFtFZVezYz2pWk8ThqfPQozNHN6ZNVAk5qIpZzPAe3E0Ocg//JB0tB5/mRDhpru7ioENXpPP";
        sout << "VEh9NZ9+MQKwbEt9HSujQqFSI1MmFU/9d/KWDibLtVjRXMYoKeZzpRUoWYKXqtCU1UbRxBuRFlkA";
        sout << "8Q2JNWk5GH7/VyERm55ZNcE9seH1XgCEA+7ABjpRfRzwRwl3O+kl237yDdfQuHthBoVYkzAovdAD";
        sout << "R2Daq1qwY+e6oMH1jqvVl+mtbwELfmDIo3RfnJrKsh/WigkHxF3V1yGqfs6CWzgAZnmNRhzZF2CC";
        sout << "HzKmNXBV26wxAiCcWGCnKyafqPRvtffKKJDD52J4bjE1tvzaETRp9bU74/PQjJh0LAt8zuoNfkMq";
        sout << "LuN6EIGVc3BJMdIF2/htuVIJv9O93xqel8ZJRBar3oLfUizP+bFreHjrvq0VK/3a1CcEPMf9ZiwO";
        sout << "qLh9eZnQaCOw18H+vq1W4aDJcEuuzKGF8p7pPXH8FQd833pSBkpHE03j+mZZBnLYZ88S1RPu1gEf";
        sout << "3g56d+reUvvfI/45xzCLIUjJbFcChtzRtFRIYWmnf7MoZ5yPBSLpmusEN03+uBgPVxgZervsLHM8";
        sout << "CPVi6RhVIUCoWTBo9z3qS5vEELTug4rof98q+VqHA2efhcqndTqq9Mi2FlpYX4N+XquJXP3CsC3Z";
        sout << "/r6GMAApjV6v2zjYOgyua1wRzXs1WnANdN0BxXLyH33RS/m6N2DgvDXjDXLbGjpD9IFQD4NdDWbI";
        sout << "G6pCuAAZIY/TEDapKWAqUrkw7VEq0IG8fEC7sgssRHJT8eHmZTTFJ9CSML9H2RyotIefv5FGtUIj";
        sout << "AL5h/zX9YmZKFyqQUvz9YQ170MX9dR3yRZtYCimjhD+RU4QmyQLV6xLhM6OeHXmd80ZHSEJLtvuw";
        sout << "Fpq5A8vrXGov6iWItu5O0P7Q9VydpF+laJpSSHTRU1mHDzHhOgxCcowt6NWPmWXdN3bP+435pNq+";
        sout << "2E/Ts8J4gmprIMPrR48IyHyFCLoL3UBFY9HfJGWqyxTwox2rlsc6UODbqrA0TnD/BEZIkyUquOl+";
        sout << "Xqgaj7Y/QbUBP8cgZUmUqDmT6x1DF8jgD2auk0K3PT7/9nD7exNet8pEzINrGmf+vYa5x7bFYSUW";
        sout << "5xNuJGxBJHSX1c/UlxK0xvGVOQ3apGg8ZeZbcQkhWcDVKH7Wklr6eiW4WffphGacPvfzGVJgVUQ4";
        sout << "gif5XaTDki28nWKWA2062fLJ/pihJVHkdePBmoljcncaHEjeAIdc7+1SREaB1qwdw70GHgvbD/MI";
        sout << "7FruUTJDzRZCFTSBlqERaTDeZpxltHdM57gybCngXe3UuL54Ivnih6i1/3R8pa+5VCU/ZqmTZAR3";
        sout << "2o9AEpe+64PhP4kuZ5dXqUhhVbscLv9aVETxQMfEgUA+cU1mB8xhegOAIzuqrydthsP5ACl0xLyR";
        sout << "c33KhEjOzx/heC74ONrGCN1rwlF/EVQI56a016SkzrxaCVNqfRDGtyfftzvCZZSGOdPofCJtRFOB";
        sout << "QucYUQPMnLxvqTg9wsnBbLiojgdPIAnr2BTM3mjpjsCqggP24fw1RA==";
        return sout.str();
    }

    // Returns Shakespeare text formatted as a training prompt.
    // Decompressed format: RAW_TEXT (formatted prompt with instructions)
    inline std::string get_shakespeare_prompt_compressed()
    {
        std::ostringstream sout;
        sout << "UU0b45RDWgwOJdW8F9arj/rV/UunTsYPc3jLQgATdaUB7qV8JQZbL78+bE36Veg5q0eSr2VzMzC4";
        sout << "vLc+mieg70a6gDRmf4IEOy5CWtJTibaIkSjTstGLVRrc9m5JQx5T5NT0Yi+9SRtafpdxARBDNGTx";
        sout << "M4ZtgcV+ej+ivG66FxbTovjYg8/J3kf+ckkTz0/D4vOt/dmT8W0tRoMe8hYowz7j1mQjYHs7Kv1j";
        sout << "w4AxjOOjakqehRORqujydd3A62MQjbLUUs/6HUkNxMPzO/nub+uvxqtG3uXTTOQ1MFR4AsqAbF+2";
        sout << "lcZVspM8CKwwNeHEtQA=";
        return sout.str();
    }

    // Returns compressed internal training dataset.
    // Decompressed format : RAW_TEXT(plain continuous text)
    inline std::string get_internal_compressed()
    {
        std::ostringstream sout;
        sout << "QmkhNakiJokjz9X67ik9R4tiXRfyJ0qpPsCCjNk0/lrwHo0niflnAjdaCwx6qr64oSP5WprkZ4kC";
        sout << "dk6cQByAmEFRaNo10ooYrItD7gm4q/BOs+VZD4Xzyf0T7id+I7S3I9C+Gq7JbNxMbuM1zykhGfi9";
        sout << "FTz/3V0AVDWBDCfWKTPSk7vivWxXfgiTfjqRx5fTqtvXzMmtogP7mpwJlQfvtFOmfbknNZoaqJMU";
        sout << "CgzcOh0wedgkGsqiSJ0VSouGKRbz7mgWXQRVBOfb3tvNjCVJfQcfWUaoHIWcRIhHmuVURQVktMUJ";
        sout << "C89TvnQEci5s58OZCHiRhqsNQjOZjTmlEzDAoEB8PglVG+7AelLSjJF0qXXFzD5SbAT49Ndoib+n";
        sout << "Gx7fjQqHpNRLjYHcOvsL4W4LOd1XXU06vYb0MqoqAau1Sd+IBNun/LO103p6sfdL9Ja2BF35oxbC";
        sout << "0VdV6Ih/W1JTLmZWiPG3UZAP/4OZuOaC/D6t/pDFQodwwuOthUm+nd/XCMfsCfkbl59YEBWa5/Z+";
        sout << "xSQCJjzXVvO5YQ+rQw9RnqSe/OsX1AJj44ZHBEG45uOlRAHr321WmauuZ1+acvHniWzrI81n5Edo";
        sout << "EhrYts0paNEFTaIzDVXTYJl9yYFNGJlEMEo0ZRMiu4mHIAOFYLXwtFpFbaP0/fjaTaRaA+ScHZj5";
        sout << "aGoS7I1ILnLxB3R7jsDuUjQeRDtW0FCVGnU4ukt25pyA4oLHgy/WHb2/OqgUWCfXt9D94piwaSmW";
        sout << "dMw1rP0BKNy1bWOrID5Lm9fVj4vnBgmz+dut2KJy9apt8KJPNTl3/raCyLplmfBSyt18jAIIyFGl";
        sout << "H2QCj+ImODsyjLZRTnY93P8Cz9Cho6p6hdN5Pue8P6Y/FY2kzDzuhE+CghW1SzYgSYK+LkI/62uj";
        sout << "DyxZ2/J8+HUtW74GC8+qTPtJAGIJm1HpGMvJ3fa015WbjIwWHpiEc9LfW7rJmkUsETmZtQ2vU1y0";
        sout << "P47NXESGsjjgVX13lSOLIXyanbW83ylQ4whQK/FzvbV61Fy8zJWs++xH/jhsjnr6QaM2RNb8alPc";
        sout << "orveCjAMrr9LADF1I6G+HsvrRqvRiLn/bYiN9sFuxDn/IacjKdJBtGCQr6B0I7AQ8a3tabUu2p+u";
        sout << "InEJVPhfFDnKR7wOJW03PCa/MXjNEw89GjBo67YZ6MqoDugP/h9tYJ12iZVB52M7MEz55Pug0Vjm";
        sout << "qVxxlX4POHxG+CVIZfK12E0jxmyVZ6c2OZEFaStspe1Mm7zhddX8iGTrzEdMyUCOb+mwLzFBt11s";
        sout << "v9Fmm/BAQORw+KP9SVDKMQT0Ell+9ZqQ5img26X5ZmC3AS5JCTTKkF+BWSV1H/dKpqA9XuEjNl4u";
        sout << "AQqZ3W76ErtYr5ToO/FnbbunqykX4VUOdylOMwYRIkOZNx+4qMO2xRvFABx4mRecSY26Xq/YXo7w";
        sout << "G2rGIWilIo5WCj3miiV1ghHoMoWidStCHJ7TnNB1F7C17rjHHHP73gGiDa3GllAGQFgun/cBxkWu";
        sout << "dijdNYcpnljMz6+XCnVolwMiBqaskpFvxVt5j2sRr53LmGuOWNeFJNo4QsGm7WniCG8/CT5QncLn";
        sout << "BVBKeRT1hr4MiAqUarwG4P1G/jPvmYazB+ou+//UbIxXUyuF/fBlv+kYNK0ceYMwqJPNM7kVnU6+";
        sout << "H1vgMvNKyUB/FXAT/w57KjsCE16dCQZaUkvJ1Po4NPm7uWGQFEJ8mEcJ87iNj7gHpSuvW57wxirS";
        sout << "jfWjKEniJUL1gzaYb1cIkSjGaxbbAG8ILq1ZQbdFoYY7L4L5Gd3iAW9fRS3r7wqWzOSDo7hUmKYz";
        sout << "fPyCTXJzdfDQZQYdD59HaUpwotz9LlYOgJGzujSEk/AosdkBnh08nW4dEdaKC+qlJqSBl0H8RoZG";
        sout << "rMxP7kGxGxi2rt0EAMcMWkR+B+HvYT9ytlja1hdjdpR+XLH6QDclyISxW9fm1U/NfoYnKyJt5u2d";
        sout << "rSo7GZyvkFQwM+8B+flLxEEiCC6gxHefSq5nF4oy8dAJu/vfmqrH+MT0DFWnD3AzBrgnZZhzcPbn";
        sout << "JxpfFlYc7y9d6K0T3eMmr71iAiE6OFqDhyiMKZgwCDUox0igElaskKiDbHOjTCpnQjnqrtTMIYDo";
        sout << "VhBxkDnlnAyOCbRhwkIJbUwKJ+bOuNZh1AgebMcLXWOyr5VYVUenIed74ww070j44cmusipkUNGl";
        sout << "0REXQmoVQjq7CanEMTkdlxwMasvFk48vNzAAANdwJo9ADdLtvvtez8OcFWC4e1bYsqgkok3gwJK1";
        sout << "EkYc0Hi8YN5MyE72CdEfgbOP+u06yT+o2sbbc7iQ0bLSzT4RSNYoBxvHL9IwJ1piH8zuEcLOdqUE";
        sout << "DcfRmLaE13W+sonUYDhXwsAW6u+ayg0kE+NmSHgtze1/4L7MlPDpd381tX6n3LepxM1b7WChTVqy";
        sout << "rLuehGgINT0AWvBXUSSm+g2x3LbStC7JFlHc1EzVwaBqvL4cpKBkHBnCNcLMzzsyp+VoiTVL9Gwr";
        sout << "NRdtMDPsec7isy2ZBRQL1ithW2fIM/qy7uYyjhgVeBV+Smr78QCP5rR+iiZsOc2JgSfIYMCJsefi";
        sout << "5DMo+ydcXz8NN/eMmU5xoQtRgkzCtqMzpu+8L3bvqKhx0sGwvluwMucg21bsaY+7XZMD+NFPNCLc";
        sout << "2k6VxxsD7zcT2PGKPgOZnpBNU99Bkl6vRLcjx3P5czVBR+Su0FmjbMLkx0T1u9TEe82Gk2/273r7";
        sout << "0SZ1p2o1cSByQrRb5vqaD/mmbZWvqvLEeyvFfzVxdPZJOzeAcU8HlPvg2ijRhStMtD+IBj6SfwqT";
        sout << "FlSsMi2J8tFK5gN3AmvPwey/Mg0YoCZ4jqefLLN8S5AElccx1VNwqi4Buy0WlE9dv4FGXZvH8GOl";
        sout << "GR/o8f6O0/WCCr1U5ZvHvtwojtqk0LTRdq3IoxnMCw12oupSZy6TMeJzhxFBOArLwlQn1qgt5ZFQ";
        sout << "ZiG6D3MlDWNk9KsbPyJkXG68m2yzDowsoH0W2nkwklLtvVOwv1ib+bMK5b0RveYEYSHzfH7p+AJ6";
        sout << "+qt3hpIIaPhV9m3kOJKtyo2wFwipEaRZP3w0WU/ZYlJSO74LY3NrpuRZ2iEWRS8s8O5OKvUYG+9v";
        sout << "+zOdmikpQDpCQsONdy0xW7A3cbr/NLVRp03YDnilRYdcAu8BYsr1mePIbR0Js/ubSiLo2jHoN3rY";
        sout << "iolFkGTMAeZSaFRj8BdLBeCN7d+UR1dLPq609n84Q74gj0GkTOSuOAhyQqMTSIKgg5Y/Uq+2LrIR";
        sout << "Jf0rIyILtO4MPsGtshuTvTcA/nGDWQ0Y4OnY2/S9pc7g3Wz8CpLfrKvGIdHQk2HbCbo4w+JmaHXF";
        sout << "vmSFAs45PEz37Cwk/MOELueRNchqBZTv3SBjLXAXjbj48jiglR8v93PZ/kBo76hpuNFhwHwjmWU5";
        sout << "Ukt+kq2dMt+JoPpFO0x7WlEgUlwJukBUESdkl8A2FR8b6DhLvx28kFbL6ffQExhONFbyQ1P7MUtt";
        sout << "1a21Jif+DVsY2iT59oGm+050+plGMXAi6egUqpLAvsm/JpMCCsOQTwSJAKxI/s5FlpIagFdFJ1pF";
        sout << "N4ADdruTEeGiCU5NjZcOa/6mcbNLG1OF73SV/j+8Q1vJdKk6v28YVMV9OsX5iMKtI0JlaiHYLKHl";
        sout << "xfN4ce+WnVWVxgA=";
        return sout.str();
    }

    // Returns compressed physics paragraphs dataset.
    // Decompressed format : DELIMITED_TEXT(segments separated by "@@")
    // Structure : "Paragraph1@@Paragraph2@@Paragraph3@@..."
    inline std::string get_physics_paragraphs_compressed()
    {
        std::ostringstream sout;
        sout << "VFTv39I+uhO7B39E6TTrjoq7i7XEbd2mHKlQfE65rwT/qMOwRzcz1TtkmMvOmSEjYHdfRjkUTU1F";
        sout << "I6oX9UwmpzY0RNYk4dorLrCEgD4MVApc9VXdhx7ISjWB0U/QuuEKr2LSzwf1xC8gjfkiVoCJ5Nc6";
        sout << "5BKVUHeCKF73sc/ciwmm+coR6LgCOcJgRIfKbuF+N+JzvMjKFMiaZBMiLa14UJn5AU6BMXRJvXfs";
        sout << "TomQaQ/YIYUXy4xXqSdc9voYdamkeAw4IyrnQk23+gYEpWBmCDCB2giVQfaOcTvrrL5L+uHzQHbW";
        sout << "VzQ8CW5yD+LHUTMeLsWHuFPf8QTDxKMi1YnLL0k7yu3VwGvrzGvxWND7W5IQrOWcsVakz3h1i3m7";
        sout << "WFh6Ln85R1KxmE7si1rh4vimPhGyVZfHpRVQBWSZSEC1P0F/kWavEehNZaq2OwH1/Ov3rUzsX/mw";
        sout << "pAaW8/d4MgAJRF7uVDI4mAW9MZIwcoHtn66WI4TlJ4PY+qyvZp4T4Jw7juE2Lp4MUQKsEH4e7sqK";
        sout << "Lc82beXbtOd9MDTdv8++RUO7JvsL2pGt+q9j6BLVxtub4ueyvVy57mtblJ8BjyjPO1r4aYuf7v+q";
        sout << "x7gUlLgedrsQhBqXm9625EpYPoZ/KKpv8M70u8OrxZQ+l5w9pliisItqqFf1VcJ4H6+teY/ES8fE";
        sout << "OR2+yJt0n30Fa+v8uGIVJGE9xZsQqZKE2nGKl16EbNAm5D48tPiqavcidTwRE6LFT3XuHgj3r/S6";
        sout << "+weTMkoxXyCrYZw3Rtfc8qsAE70UzECFEjebW//Qg5o/IUnMMdu/5H2omp/gelvXr5hiNlHKADd3";
        sout << "rTKta7r3OV4ku38TGDFXYZiwJoYcWokQINXF2c1yw8YZg6g614oWODLHCycgS9cYVq0snvuj+qIV";
        sout << "Dp/aVzipe6AAfvrSWUy3QFpjdKEtFq/HjlHBV/lWi4ebzkEES+5QpPHyRjXbUOpXb9Q3VuKKqtxS";
        sout << "XhOU3n8RIqy3fDA674wNnhl/ft6qBO5/oLV6T/+I3u1z9EzNgbSw68LSwDwKAH2pQ+qJabmQYowF";
        sout << "GmH+ZdjTGxrNrVH9TJ3tIhfhCAZYMv5dSxnJDtfTDoDhERa7cdu4zV1YhGItXflpr1JxMKwudINL";
        sout << "5GtwkiI84zQaUjElnYG7kGwmp3Q6AUyIWm9OkWj3HUZpz4Oc0469PfSOBIQ+1BMebgJIhOIuNBav";
        sout << "AEi8sAqy7EO4txN+sdcwARXSWGAcKVGFMn6zMecVwNdayIN77a9ZpWGnErVY0nTtY+vbR9zkUvT9";
        sout << "zkK6Xs+VIRSDkhASweF8yzJrw/0qOPsLK5lQ8h7/lZS4aYxbfB5TghLcZ8qrPGQs1AxToDjaEXQg";
        sout << "mJvX0qUvISHMiQRloAlda0U/1Z7Xhlek/+NVn8bqh/yqG5hPTPp25DOwIoq0qFgcaVIhTMebDJVX";
        sout << "sVrs2/SP29nknp2D9ZZcRH+3RcWNuqSy6lG46PR7Qs47Vr9lRiSciSDLySofWMXtllwigkpllimY";
        sout << "CR2XPNbyPzUPtvlLyal2RuW0QBQ/l+h2KLDen8GUtl9S+Ivn9b9f07L3PqXVmf0RIkqXJyMeVDK5";
        sout << "E25naL/xilCbyJX2eRqUYExNZzx7jOWi22RBXuKahZOSlsWLsBkZ5dNHjNe8ZVc2sXAjT9DESP8o";
        sout << "ml+7/HUZpcCO+PAbwWc+hHAz0kylGGQ0ZWimIz+dTgZTAKhX96vASj5Y6K5kBHiMTD1gMIOU+gSD";
        sout << "pQ3ytl4qxcx91Of/IjmqUsqu3qIWQJzy//YOxVHinqBQQ7XoSz3/xcXGUC3SoVLJIiVMYU9BwiBt";
        sout << "kkCns+pXCSuLIWf/D0sYmBJNYz0JGHQs3+5xOWj90FzZ2eHJ/GzPyXPYhaZ5WLXjJTuqRLnQ4U/w";
        sout << "j7TPaKDVZJigf+0BThuVtxnWc6u+GOblBe7Xh183kWmyTdg1JHq76bbKVXiEikvuW+in/vZ9RrFt";
        sout << "NwojUdqNGlx4aSAgpAQuSch0v19QmQe9mHCeJgSy2y9+J15/aQC4EbOl0br1g5rgVI4dva5s+9Ik";
        sout << "Xm9T0e34qAK6qzim3FfNH0rks9vED7b7ZdlyCG21Z4+O55iDEpZq7I0rrjiWBa8IsDnZ/qE0e+iW";
        sout << "u9eo7lAeVnrMYRxgQ5AuwSf/1hP1xDDZeKSdBpdQ1WMtQFkY9qwArX6Xd8lxX+i73TZ5sYDCc00s";
        sout << "qZgivAbUSSqFQ5m7k0BiK6HqDpJ5Efh7537wC89MLeB4XZLK2+DJhAOL0c8XLAG+lBnwKmy3pjFX";
        sout << "PvgSodsb++g+ue6OTh4w2gBLS0tXiUvxNoAV5rbELIzlTffDup96xdHoWIsoKolJgAdANc4HdGXX";
        sout << "171Sp1tiZrN0gNhC13F4o2VhrYkrSG2bgzQkwfOH5ugKOl+3sfWNFT9vh87HGXbvYf02VEQmk6RL";
        sout << "QbpbHn47RnJ0PAycurYTETj5NGdj6HY/8OHSOJOf67JyS72GdEvseVuLZwp9TVUzCjsDC+IF7EcH";
        sout << "DbGD9x3ygRnvAYO/nQT4G2MBsVASSbn5Kt7j6xaNNJ5iykACLjGNqip9h8qKSGVM2ouUw6vW3iaS";
        sout << "kFY3vSj4Q3Ih+fHNsZkr5jXNGh63HVQbnujKAFbRcGpqds2Qt7GkrySECOUh26YZnhVwz3Fs1Km9";
        sout << "pHd6v+Sa7TT9vv2qft5RgAfUZZWFZ9J8jhv5hooGkGSegUP+HI7X8aPaGYolYToOgn5dEOS+E57K";
        sout << "HsL9XUjOj5w20IDUG4IWmMlxXXMh+QRW1favlqjd6WIEQoayqJ94ybKw1H6GYkSek7IkDQflWA6y";
        sout << "7yMlSxTGhN+OBHmjXp2IkG3MEGH+0mPEeOaoV5jxa5OO0uLgcqPg91kFOOyU0BneTwof48l1IOQQ";
        sout << "6f3LUP6swG5hWBX/6tSFSGCOuNV94eCvfAwtd8hwUzgMVhMMxOGoIJz7XvSI0bYk/Na0BcTvN+d4";
        sout << "BGDAg8jFl9BFc/1zwl2PoMxx/IVmmRQTj7zUeqQfP8cJ74cWpxTwE1Gduq10d+ZVzYbUznvWpNhv";
        sout << "TIiF9dFOp6K41p924ZUBO6klkVMSjiRt3pBY5jhwMuqTOTDDCLLwF9YGrOaipfhHZ9tqFhAx26LR";
        sout << "YTJTyNyGZjoCXzgvdC9gqV19XdJqnlNWdQoOKAtVVgjDjNTmW8p56Hyjp4GSJaFZTitD3OU/ALFR";
        sout << "5Tbtup4p5Wb2dqnudkapXTYculgXJfKreQMedilMSss8x9PxLd4yu1ala6354I9uMRvNNiD65r1Q";
        sout << "21vDCLrwQSnWssQVe3i2MqUOtr/zxPlMWbCXrl1F9k1n782ZRGPuxsasvzq9JdiupGOnetPqA5sS";
        sout << "pUpzQG7zmSJ6wR2sh9UgEeef98BfoxmZu8M8rFNxs935YKPCSmZYcK04vqjylKXjEk6iwg4BMvPT";
        sout << "oPhZYULv9NMuRv+hA+yDNP1IIIbv95Co7mcuM9L4spuLTPNmtzivrRRjop+vE3Ha2hZHS9/KtE5b";
        sout << "yCVIh9BvDySrrNWMSPcc//xghOQ/WGIIotRgYiI/CBU39M+r//N/XGhGM/fsKYdM0HMc+r2kTsq/";
        sout << "F0wHrL8S8KA1asAtCxTVi/NLtWlqGG0bJ6mW+W3Pt+2M2apmotoLDEYYk9O3W5uevUQnAx1MoS4s";
        sout << "3CdauX+/0wVAkyp3YFQ8M+rcUUMBBUVUHK+36zeYyKi0EkIsuOBRQfhhqJx1V3kIml5EkQrG/cNG";
        sout << "KVzvEb8rTcMvOQ3oyds/5V9/tYL0bwVNp9LMhM296mDaa8hIHvp82ytZOGWXgcPn3kjS9xz+EgkJ";
        sout << "aUp2rG62bVq+xMJbUXnlwmICquZA+hLQVvqw0N/L+dmUJhPzBmnm2r8NeSimtznTDvbl4KhaZn1y";
        sout << "Ny6+yGB8KsASlcCGB/hHUH+jU4tfe6CpvQWOJC6jFEsEPwK2oWehPEkW5vJ3rp8Ym1nWSDwVoEBB";
        sout << "p2992TeghzuNroWzSCQDOzi6y8qRbgZNEhXkLrUlp2G3zxdZS2twNo+n9hgysbxVATaF9XzOn3JR";
        sout << "So61X3BAzb1uB66Ano2+CrIgFg15Vwi4+eyQiFJMneLxB2xGZqC54/vbWisD4CFhG4LeGWCYy+DB";
        sout << "ZZGf7LlwGaWPGb0WpgSiPZ0cr7K/BG//0ExX4ab081t/W7pWTIe9P+mNpNegm/uOlPjF9C4ZIe4b";
        sout << "AOaC+095A7jKb/oimDJGj9YcuvSsLW35u+7v7Gi/LHdJqCzYkuPKO31iN/PIpbxoadqwRCW4t1ZN";
        sout << "OAUPkLzFax7YciPZdLHUxu7fVm8vCQjgpzUUCi2zhUE5N6fM5jo7AY1Oxmimd4ItB6AUFpaEBpsw";
        sout << "DR6y8vCKmnGB2Hi+CvUJc1C6r7F5FV59uflO6kveyCeRg1Htriyb4PdT43AAGti1uCYR1DS9KlJL";
        sout << "PpYezsDi7a8Ye6LrQ/dTnewxxwCx0TuwAM6P28Nyd/6ePLSANdYBittnCzZNCnoELv8zKmvM8cTt";
        sout << "UJkAL5rcU0qc+qf8NZmjmouc/2/Ah1NiteMzNK+6aDL6uerChBGs3pDrHMzwUU+Lxbj7oBAIqq/w";
        sout << "X9jqpRyz1dRvs9YvSv6nEhrPaJvOXAReMsy+4rwNj7xTPxrIWvGEULMoZSvJ8ArWOkW7w4QhJWHe";
        sout << "A8u76KgvtRs2+NOrdWGIIk0I4hwjno+OsgQzBAnRKsJ4ZwP4V2QvlNSxlcPjrnpdOfASZ7hPdefr";
        sout << "J159ugCN4pbkLbj5PFt0jL5RUq9Je2Es5f1FCsmLnZ7yAZOVr/kSZWjS2ma4PPBx7ITfBP0zaK58";
        sout << "mwHXb3pJ65rjdzFxKXqsato0H2Fe1TER4gjUjoIr9ERtooR4uudf4FNRhySU8awXZVKIWv0LWTVh";
        sout << "h+9JYMC9H07vEmPNkOYK9wvNfxsvnBrasJvnTMkmzCEQqQUl9sfIOPAJqG9B71dcP6dNv36fc+qA";
        sout << "YC020TDK+1mA2WNQ37Z3Q/IXNnldxDL2zYNvoeX4tyakRIW3BOVD07K9U+3URuLWT+Il7CJIOerK";
        sout << "Gkx77i48LITvzSAOxgvEI75WsJnPgRQO8S0y6wNM7EE8FNA+wig10iPffQg1GcS0Lk+SEHhV3QqB";
        sout << "4ZfNcIZLN+iu5QYQrcniFGcDg6R3S32qloDP+62Z7Db6otZdXYB2+hFb6C1e8k48D6L/bZAfcISp";
        sout << "tLT6Y7ICFmOgicsAdlVqEfBC6FB2J4pEAefZyiaONt1CdhlHrzpgoe2Tf1HSNdjE6SZ+eSfuFs9E";
        sout << "QGtjhbHkMNnbss+sEO7HnNoHBWYG6e9Nz/ekHQ2CI7y4jSc2WVfFlueOAuITVeLCNL7sUDtQ0rrQ";
        sout << "bihdoWVyzobIQXgh9ifzNYGdu2MuHGH50p6cH+o1Yn2KRP/1iha3s4SEu8XgVfkXuaN6qNMuw4gA";
        sout << "W8DSPs1OjeisMTTxZ/AupjSIUq+bB7EhlTSTO9R7L6WEYhZ7KKWZqnamzzfl5/fBrmj2N7fAdMlF";
        sout << "I+73GeqHepoBjWBSrxCATJ6Tnk6u7n1CXhNCJlMoiy5kTLXV2mJi80OSU4rreU1U3EYS74v4lpjQ";
        sout << "M91W8v+egneVxVk5cizxV7BZi3ps10a5QPxbsvclczC8gaJmZKu3G0OtoJ/WDPkzCLEHuXRU+h0M";
        sout << "Gxxz15NbSCR3NjURjjWslY6s8gp2gf4qQMfCwAq/m4idVhsswOXVTgoD9Hyx/825Ry3kdAzaANNJ";
        sout << "LRi+jVTxhAW/3VAGOYNXNsSuXqWWgl+rJHySsZO1jyMFru8Hd61/H5sYJZucj0KiMaGd5nIEzNfO";
        sout << "LMwOJdfFReEyiv/BcC0TLyCNSpfiflKSEhthxRe7XscPnzV2oTOx8eEFnFK5OU/E27lSna49n/r0";
        sout << "hAv9TeYNCu6+Dq9oPPx6D/svHzmNBUuIxOL7aGUus97NHB5k/81rx90PImECdmZKwwCdV5IDxiHD";
        sout << "T1OPyQETyDHZizXLJ+VCMiVIGWBBVulqzqKYwlicnxjJEAUTy44lzbKKFpD7AXghxzZSAO8265qB";
        sout << "AudGqzOckHTt11bOXARXxNm3r5yZETQYWInSAugCy7kdqBTBYpU/pWGJzHR5KziWVAtMPMhoBHfC";
        sout << "X4PnTim/UTqC2j8SimAIDkgGJnbDTLCRuRxAhfvFdfjpqlwUh38SjJuddLZben5oNhlsOOXYv4cl";
        sout << "20r5662iwcDZkrELgwJv6hMQ+U2w4zlXS8QDkvlmMzKk0FQ1bEFOCZLFpC0si4muVotrpYujbo+T";
        sout << "Myjw6ZeQZenN7gVFqVsvo6Biu1YyR65v7HTOSEd2BpNDpsFeIsLwdoEvAG35JWwXCvSM/yxV3pxm";
        sout << "f+f5GX3JBzzlLbp5vVtw4E7ambWo+opyCt8EE/ZEyiJfxCEWrxqVSam10Dk0LdfFQvpksbhpGcXP";
        sout << "2S+Cbhe70m44ZJ/w20vewNuGRf4+vrQqQ9anGiX+grsk9/IszVWqRot7UopquR0zT5KFFiqBZ+Ql";
        sout << "7OMgCGD9O4j4LD+csrX8V4k78qNc2HMT2jaMvnIMu4r0PNIS4Q6k4YZ5W5lO0VdAM6vq3JaNj/Hk";
        sout << "0RggcQZNCXPVdk9Vl6lRbWktP1P2GQaXZw2ZOQYm9guL9FeRn2grbo7gGZ8Yq86tqvR98ZnITIA+";
        sout << "+ML+BjFjMQTnUbQGYIyHV/aepIvhwy3huF/jXbD+73kSjC+GzBLQlJ/C9D4AuhYfoXdMdHm9n+19";
        sout << "psPZynDsaCN1I4Rp9SWSn19ximdhKlAQwcKWtCvDT+o/e7F9XTevruIiu6XbootWsgEE4/72z3GU";
        sout << "WDnpPl1nJas5T289vMHtYrSfUAHdzK5IVSB8XD/u10iq3jcJkoUM1OvEelb9W8rfnCQH6802bk/6";
        sout << "0PKp4BpLkkDxBB93vrvRKuqaql9pGShKIN7zQ63XjhEhQGgaOOmmphD2z7OyQCbu0CnalHJEpJMb";
        sout << "nw+TLmfTjfWRvlZaCWoMaHzFSZQU+RwXAFsjUBYGDipZBo1+2MWkOP95PRn6DntKZ9oQCCzGeyxV";
        sout << "zxQJ+YwFUtZkCQ5ieM4jsazgEzJvfcTMSZwfQnU+XrYyz8vexEHyJyJysbyVhnUQ3k1+QXEcX+Sq";
        sout << "i9PuCLVyWfInnEZ2z1rJuD9ZXQkOccnptokVolYZM9L6JDEoCFYn1BwJTMVJ36DtQ21r18BBmTZM";
        sout << "U+cErEZPV+AmXhsIWmZKboEwfmAxvUY56xHhLBjjeMJYVHtzqhIysNR8hN24a8YqV+ClPtLFI8ay";
        sout << "2HFHoggHgJkJAwPLD5M7/Au/8lhNbfZRI0n7/uYoJ6L3szgauCgz1AgOalqWMzB5ysP38AzleuTK";
        sout << "AQkK1pKDM/NQUJjhV3VWCUMHhIBCZz/JnCN5SfdvHunvSCkOMLqB/WuZgyIoo6nQTPtHfGhhmhtX";
        sout << "RINKMA3ODPa6oAEhMbJBtuMoiEAbFvfYhWsoqTsPrF5sdoFO26YeRcLgkKzeGNok0IC+yUgOikZB";
        sout << "BueorvppDLfwYS9hyptTL0ISLZVf8i9s/I2cm6tWIMDZeXzr490Qt6LqD/Oo2585hpi/uo1pJmTP";
        sout << "MO3usStmf+mUPJbZFgmBtgRbzudgo7lPDnKU01UmO+QALMxiMqYLEz3zX5f4UFi8N1or2s6Ekdf7";
        sout << "iRdMOElIRbFU78auL16kHUtXJ9vyV2sXfyjLmnkxBfeDgn1tHfa/blfTshbIVHbreZTMDkh9C8rm";
        sout << "6a8ly7hBjsBgjiTXhz8J9+3xCWx8nzbH658hKUFt4JaHEo201LHjZw2adsNIdSGixGKhN/VVjG1+";
        sout << "2oESnRRp54W8oA9KTcS6CP4NaXaCxiBofkK9bhwMgjq41q3MfrwxW1S41CbVL2d32edX+0BqLEdu";
        sout << "gK1hrtStI/6uwJD1NVsq7z73OcbvC6XuCFhupS6EeDUIP6AGWj0tMPzDAhp6niyM31/brJKKSPgm";
        sout << "PS3vHEogylsDKI/dY/qVGaddtASvgbFT80fkdU6RByO81ehINh0y9cWphvSqYjnAsbCbj6ZarQZO";
        sout << "0VDOnAgEVtRWbt1C+AhzUxXkhkwt1Dw3J+yE40QQYWPIcZv5a7if4IlAm50CzAQAljW+7J4ptNBs";
        sout << "NHPkY3erjqUWXXM7tK3Kb+o/WhPNt20Mu/JDki5Vt7TJPBGzZuNDPsh28YNG7B4hMszooyPTVG6g";
        sout << "dXGOG1YrvbNfN+QVykhMMMouBa+PzwGavhEPg//q1NuTPjnbyWQumHBLCRhSvYjiAkXSyfOCBseM";
        sout << "KUGkNDhZuoE+KAh3NJIwoUfrAj6xGcKxlRtI5Cm/0ezhS1dcH9dtnFhXWgAxuUSikSrV3nz7ZMvr";
        sout << "MsV/cPqwm6byXRrMPMnoA/VWOGKR2w8bL8rFdgtNhb6PNI0572VGVJ39gbcOrsSkFkl0KPp81/Ga";
        sout << "szShON/k2kQgMwraqtJ46gNJpNsr7DoGK5OIrGCHez6AE4J2cQAKb5V6xVCyIm5bvu3Hpko9v3SZ";
        sout << "ojt0oOXAPeA50ZmC4F63tVQVTOh21AkBZdhZKCv8iHk/Vx3r919JxSeBM9qGk/sALpIqKPQ4AJPV";
        sout << "d53ZHisUz7xJSVTRFgSih47cb6cFxLeq8qlmDWKbfLSb0frSZCUV+evOdPcywP2N7iIYWnvtjqNk";
        sout << "rUgI1OFGDddhrJSDj+meoIy9RuXfXktqRpKCQ0TUKmkCzYTteyH5q1U3LCMIjAS8vNsyF7oLiLMx";
        sout << "2VrpjrujKtYE+NUTZ9LtHoFwUrd9OegU20Y39JDwvJUQ4cEW3Einiw0CGTmx9rYpeYCq8OxmLhCV";
        sout << "Nb+asKluw+/BJ2SdEK6kR+6skPS+ZmBc85KU3PLhxiMxqzAktc2yEB6tUFGTqcoFEbZ+4eVRePyg";
        sout << "3l4QApAQla51xg0uxsMfMKxeqxlCH8Eb+JbBVH8eEs4J008PLRP7FIe2SmkzCwxeX4Bt0rBmgEVH";
        sout << "u/zb3TVojYOWkF6jPSvS3Gt4B9tf2TEzeOYP4e8U40nhdcqD0B4P13P2qWTV4EbKGX8mGP5iIbTJ";
        sout << "IGt5NjzuCD5+nCzsWEQHrM4EZfEQMeA+Ki12tb93RkvzplU6RV9sotJRjytfyh0F1dqcaqLHufjp";
        sout << "xB3d74hp/w94HoEq2a8DslCYKgEMVidrITZn/2eoIAGVzsIh1dGrRyA0gjjPS9sd7Bvf+I50ZCkq";
        sout << "Ornaz5iFQKSqW/BdD0Lin6/NXkTOmj4tfdtTGcmVpwa2NQrCAK0B0PSfp89q2QImGtkABIrmhQ2p";
        sout << "V3CgAO5PIdZzy3w0hITyp2hacWjjkU2hu/LOErhZ09omtUfxEa1LAkEOzdeu5YVdBNO+ahSc0aHf";
        sout << "Es5wWmTeBsVYQUYJ1z7CyBLZtK6ZizdybunIj4ICiCcWqXiSN3tlYspMaUYM5J8TEDqN8Os3XlSL";
        sout << "WLfDES+/m7fDhL262kyGCvCpd9yRiMJvo/sOuu+TNklhPdjqXYjtNAfFyfKejsVnQYBAJvgiWlJ1";
        sout << "qa044TFiF/lSSimBf3svwSmqj+zd7Sa3QErAbM5tIzksQJegB2TNSLCuCYLgNU89cgO2VktCBYFy";
        sout << "dN7VZpsqWwTjpWXcAokH+BPLwWRrB9b2mZXBosO6UUKYf/pLDp89up8Eh4AHjokZPwb0cqH9H7HV";
        sout << "ehlmiZUjn8j/F8QYF1EZuPLB2BtnYKVFPCr+QwPdK2Tj4QAH9zJd0PI7by0SS/XQ5d2vCAA8DSD2";
        sout << "YA2fXscr/a5CfK0gRDLd5pHb9y7LR1qV3a2ItKp9azEm+cT9XXEfpsqI/x+g3x5rwev/spWQbxjP";
        sout << "lrotyIB4bt9pZ2u9L2qxZovSc1oT7aqWkCuM1rcBMP9k7hLquX0dhgiNq4/QPVhu/W1U4SIgvu4N";
        sout << "mdBbZ/pffG4l9ifY5mb9ClWWtlmA0HEwjyWx8870NLfJfYBund5SnJTrwEzybaS+CCM85QGjeFzF";
        sout << "Gpng+T60kL2u8EJqbVqCaTgVpcrnYFeQPqwEpdfHcNrhZpd2hz6UHrXio/CkxQTYsJbz45TAUY5E";
        sout << "sPOhSZ5rKZGvmvgFlUnlP4u1Y+9FMo2unS8iBQB3BaxK55+3o8qTgogWs9M55KXDHPKrZcH75jet";
        sout << "qtDDjVywOKzLRIzdyxkUth/KwaPgALH7FT5Kudmzhn1mTfgrbe4HtKmD10pwc2PE203nGBdDgEy4";
        sout << "Tgr9h3GWE27FjDGroa/7iRou5WjfXLWvomDLL8zJy8oKXdJJje/ee2w2FMKZfLhGYgeKmzDfSdWL";
        sout << "I3njQNB8UHttKZ1wCsvjoefqE2YpJNOtZRu2QqfpgNf+GhPQzBRhWoMYvrMwiN1Oue6gdwja/lAM";
        sout << "JPnyGj6LP5ttPFuN9xTtbjYRGdgEH45jquNNeBG/jfTnacdo0VXyB8BLA+OW9xj2NxXkP5Ey727G";
        sout << "ioQ6RU0Sbb/syK1C9IcZ5FuDEeIi5EHL66oiuI0dRIDjMabmhdwkaYIOqqqp2ad2WPq5XoJr8zzT";
        sout << "MCoOa8+3KcdUoluyc0v6TJ/5e5U41D0RFTxZlBHniKasK0CTAAqxbvzOeMa+vHpy9y21m6WRi2dk";
        sout << "jrDd7G3PSd9Ju/9+nCTORdPQLy0WjE1XtTOEGNHmjg8FVvLZpkwYltyjtDL2pSmDQYS5MbJR393b";
        sout << "4KNOxHQ2Waxvg6OD1v89dBkFJzP/4MIJ+06J1Ic2nGN9HQf2wOXRq9nLjrLLP9uZ9b/lv7+ENvPk";
        sout << "Mkyak/P3nb1cGv5d+nA9341g6ux6KRnC3ODniIQBsaqCWK+j1ZSYt3iL9PjNEWltXI6O/eFrEZ/u";
        sout << "RK+3U4jkBm3chL56FZwVb2flKi4jWNw3KiiwMhCvRbKWRF74VMN/8lHHgaDV1GGYQHPFBtQhVO4o";
        sout << "1fdU9ot5n+2hMPMVUi/91e91CJhZ527Z/O+ThJzn/ZQeT0MFSYAk261cPu6NONDySbyfpfAD83c2";
        sout << "H3PhfKWnNAZucDQvJfkhdQtA9YEcDuKqUaRxIOS7gLmBYf3yaRU0oqPIAB3ArFy2qtOTB/7QpQT7";
        sout << "TE7OdHBIcfwb5pc9kCvK+yQA5qr+10HH7IOleRACtrjPSxwi9kyKK54705isJeDJ6mYzVSWQMUHa";
        sout << "ChVcw8X7/jVcmM+DqMI6eTxhzoderY3rwdrO5wYhf0vitEG0RkzffKfQ/S0L3ZaB8skusNOeIhCk";
        sout << "6mB4OmyWK7QrnR4M7bdD19ghsxu0vgjG5jSs7mwafZ50gAaE3b4b6TZoxix389Wju8YQzf6GHNgh";
        sout << "HFjoWVPEaHh/h6OGD3Fn7FWHJpjscLvTnMNqMkmLRdpobHZjKjyLSSMvyZ8iEri+wFZYWf7X+zGk";
        sout << "iklQoN9cttJDehMoge1Dhe9z3FBJHshtW414yi3e3ClSkQf1smxjDHKOQP+n8+Fdh3yYwXhUKMhV";
        sout << "Bfotd1RB/dWezWZzagzgssqf3ahLipWUjcCrewu+HG+AdiSkkwYTaoKI9STC6E7Tw+LtFvY9y/SR";
        sout << "pcSCoNsB8g7T/fZ73qB0PfZ/HfAafd055/Ak5UHRcID6/iFC/OD/bWkidQfjCCacqyMeNsP+anzP";
        sout << "ut0c3UhVU89oJJ54lYB+ipjUfRxDIcHlSi2NzKHDPQHi18Z4LamSOAfoKU5vFhVzK5BI9tnD8Sit";
        sout << "etgTBgZuEqGtjg+60263/woim9gIWqtO/DfPKflIF5ywA8IfGOcKdtGWzyEv19el3QFOaW9u+ZZk";
        sout << "FoeUtJybE3xwSvm5RBMy3DwTGDuVyaBBwQpG60yi5A2Gv9s498ZHl89r751FPEFg9fh3TQmudehY";
        sout << "bZtxq4FWMbgK/8EwYn1BOwg58Vvp3XGZelcadGCU0WQcjLEzXTtretgiYQvUQ3GbGmrvPPgxNnFW";
        sout << "ScycdCkiu+XEvUmeYK+5Dicd118fnzhAX7NuUtsotRxzDUAixNla3ZD+qsniJ+dTFjzLOHnOpLGZ";
        sout << "pYeNolFIAI9AxGJWMI6seGE4hrfc9DoyF3OgXI2NXTAk4DrztQk813MZahrgt/dG2QoMytt4aGqp";
        sout << "PeE48fPOG7xx4YC42GDtCR2oRg3zpCNuUNgceJJQAyOF4HJFZKqly2BMt8OTXmYXRnwrplpiUlf6";
        sout << "qAditYcpGbv8ealB29gB6AXrQey5yaz6mNbM4JjD8A38ZESrQb3RjHKNWUDFFtKoDN99EeSn4EH/";
        sout << "V+yiZO9+TFcZlqi29gD3VW88c3jD0awYoJJ/1/xIMNwPgmZXgsrc+3whKY1W62PMJ/gQpN1WXTO0";
        sout << "1mNvQvM8YcMocM//3J1RwpDo6r9ceIKJYvzWToCOCygfSfx4o3HpmJe81ydVc4FvHSCtKA1YCdp1";
        sout << "a0e3I27ImzfoXTT4y0e+jYNtWBBd2Jwqn52n4u3kYib1L89WWpcVMLTBOZ8ps3q1KPnT1zTsQORU";
        sout << "Km/YvG1QY8gOIPa10Bhk72093armhkKMDmL+GqrI92rjDlqiRbxUX2Xd5Nx+Ug8aZ3bdESIAGapJ";
        sout << "dXa3W6gqMQOkunBDEe7NZ0P/BfaLSdgFOxW8XrWo7sbd87dP64nrJY3gWYpPlhVPHlgbjIxc+BSs";
        sout << "hG+eVdFBL2IJWV7UL0u6VfP/2fCxaFmOlWB9tXGRNc9OOHM7kUMN9KoJ3Otsm3x2RJOKXLkH8Edo";
        sout << "EastOg6EVyRRKiL2lyaihGg1siiaq5R4ApXqC1OgfyztArbLJXZfgarcETO4vnln4HYz3EhrMh1V";
        sout << "cjxPh6Bi0IAXpGLYFZoZYiaTnMKex1kCNYOwN8DzDOYR0eLQbGiOgzRGQAAEyfjJ9H+RI78lT8qs";
        sout << "rY3lmRENx0ubrGVRAGb+ixOBxR2Ab4Mo9K7ajPZA0V7Ye3hfs7V7mzVT/9IibJOc8Bl44rh10mw7";
        sout << "skkrnJzhnVF8J5qim6MRixEAHmvi2pp3SmfuBN8/F0pXeR9J580dO4pix7wuwn4kcLTXgvFzsPmq";
        sout << "t7dkQxFPRH3UdNiKXGQf0R43F90XrVQumEYWmG9dxYjZj7/FSG2wLnRr9MwGkmMXDRdPvDZ1E2xB";
        sout << "M8t3UIqvJ4BovsRwqPuciR6QdW/0sXY6/A3USQfFR1ZHarzlwvGE+KX9oi4g/zsbBT2wJA3DmYl6";
        sout << "bGeBk8xHUEk8HDqpVeAf1zfFD4b3NJ2FNo7gSnJDQ63NrHGxkBOcrZ1j9SOL2HYzNp0lL9JBvNHO";
        sout << "ZwpHPAUKPhsVSIAgKi0ZpfwKtc6NAwUggZYKztZlTAoQflgl/kUDXY1358gUdg0ark6a0rrOrW/5";
        sout << "WNYJg+53sPeWLchKUYjkY8NveEHf6EWeWptg0MptmSvpJLbt78mfrBLl6NjKQLiwHvS1VJGy6dib";
        sout << "crEHl3raBPQMvKjT4zx71XYZmDHMJhgQDQ4jP/R1bBf1quks0OB3AZleIR02QgQtmg9UehbNiGYZ";
        sout << "CbSd4hOckPIyBnO+NQt7qRWEgrG8PW6bMZoT59qIq2KiXEyQhrt9dFjNRIw2UP526O3S693QZs6X";
        sout << "O7ANkD5JFeWtsFuhuQ9Dy7O1Rjmh5QjMPJcgMf9K4u5TxJwhsPjeZkII+0DfalTo4V8uz2vmULGI";
        sout << "2j34RUp4G32iiHDNVBR+cm06oFmfn+/UEHY9g02yQwyS0KqCQ8RBUNgZevDpUSd/yDOhZf85veFu";
        sout << "M/o1Br1OruI37bB4AlAqpdMMZJ566aMPp4n1+QIpmvr7DOfwwy1RGcgDTLiSA86Ydp9HanO9doTs";
        sout << "0x6rK+Rr6PrFl6r0bWmJPs0O2hgGl48ChQc0kfGEHapb4jCv71Z2zd/L61867uof1IHBbAGPslnE";
        sout << "jcm71HtdS6E4d2csx0N8Cj1mZQDqUxq0MzthTaF91xKnrqCOsJSUQ0UpmmapRcPRRwm/z6xUUqMP";
        sout << "g/gs8YhwYt9N7YChNnulaw7kyGLANAeA/8DYOi2VTeO9qqxpXunivx3SPl+9V/HK/MXcskOMLE7m";
        sout << "85tDWIVT+xqzJawbo3Dy7dnRtqHP+qMrWhVngAMkJYCJQN2djOA4G/dh9b3ossX3jQOziZ2zQCnE";
        sout << "Y98aUzFNBCKWiE1rvCnfdKkmFpmYqPHzmhhayVI2+ZiN4aLHVqjsGSBqSbgSIFEcszU4QCR0BMgg";
        sout << "znluMkmVNy/VdE5nn6OHnb8H/3Ww/r/bhS8k6nknU0mPm7wjAIvqpK9X79i81TI8HqMOHUvICsw2";
        sout << "O98/iSyRlNF2BmzV4K9oS+aXkCMsLeYixuZjslRmiFm1yGtwkcicY4Kr0SxJoJ3/+6Opmv1cHEkk";
        sout << "Pm7aGLo6eiSe8W/4fmvuGtVa6CxZlf81SSlpM4uq/hNPkxiGkUsKJXscM3DIZYhk0Kx8LFI0XDkh";
        sout << "VlUtcEIBRmUTFiQd5Lsa7Zv5BljNOrOj70Ai8y9d5HhvTjrl3lVDsGGy/iZ2iFOjs4uqvMWX49ga";
        sout << "0KK6+3QgLlYFf31Rg7/z+iSTNTBSs388Pya6DiyKJfwLdEcNa6LNpXE9QIqLyKl7ddbw7MXLGrCT";
        sout << "jYS2YyMYRnKEDq5eX9OjHtNysMb0zp3Jspd7AwPWH6Y9NkWZIuOVFonVQUqKQ8mCqxC0r08T/Or1";
        sout << "bDP/TBG3qreBzGcEFRpz49UKoE4wjIh7H5VnFPPptu6j4MTKmDrQA5XpspDw8Cyx1anyC10pg32Q";
        sout << "7V8rnEHv6Q109K2kIJXEExbRTYrxP9KmN6MrlD0apxv/4kRwC3MgpEOTmsPuF4NZlaLr3fEtmNqc";
        sout << "5+aDXbPeQwPzD04wwY2Qi3wBoLPNJDV62RceC/Q1Zh3lOzYOG2oNwNSeF/uGqsXL/n7z+rvStL4R";
        sout << "klxuz+xfNMcz/PK/5DZckMxaAsqYQq77QQ2ch1uBVaxf8ixj/zAGZlnz6b8Sbh6txFSbicE8DozO";
        sout << "o3Nh6IX3cgX2701McY4Ziwa+8I01PkZ+4bjfzkW5vCDzoom4ZTD6sDcongZ9IDA5VwNBR1YizPvP";
        sout << "UuOEAeoHto0nRDHtQ0K7Cw+PDfoEz+obfeEd3M65BzdCfSpegPBZZmZ9wZf/JoIYCHVCbMX3CoFo";
        sout << "cmCApAJPovi7pHmC3oUpOsCssxI1huhMK7bkTA7X9S3TlAXAC8fm3aQmqAupaMkGFGrZxDZdyrhP";
        sout << "YnH9FunGlvSahS4qwT/APMVi9azffoRZR/EZ3s77J9W6RKwdpj6xS6AuZGnkFgSlMLmTGvcksY6K";
        sout << "32rI7ywetApaIOvkxdHT+XZye8EAnGsbVivIjTglbEyOz1xJ/pgaXMfcKeyoDXdoOKS9TTHIpkJP";
        sout << "BE47nyGs8V1axy5FQ0Bq59ido+4OxHaeyah1Zaz9Keam5Ap8D2swRn22rYcoYl0/vuhFBe3NGjQi";
        sout << "VkeDN00rTE5WghNuHcHLWB1Abi9tXVZkFyJJMNJUY8V1ppeI1VODKf/imhAlOOAyeAV0bI9izQP0";
        sout << "1I0KeYwNbX/+fuYl0lsUrO+tgWCOVaGjmE3DJcTwcIn9CofVD35Xfnyk/RK9kS8gzdd8DRDa3/OR";
        sout << "AYRCJGS6tv2MqNKB66ZdZ3SQQcdIF0s62vebcpd+y1mHfiWuWfS47NaRabixPAFtCtRyRY8rngVR";
        sout << "OcCm6bti+19D/DUV0T8ujeCXCH37/SiY9f+9PtCvEKxpa1gTmRW9dt4oqcaSPiTnz4lkKaxA6iUh";
        sout << "ZCxSaBmIba3hTTaTAK1f8eTY2Q+rs6ZsI5IAw901+Z9IO89YOoMwWkTPZz2ni+mt/SWylhLZ18Jp";
        sout << "LuF/dIvPPbQ18kMXPrRD/LsLwJptW9TZrq4Pf1k8m3rNjdTXBOhHdpWYSlwRZi9Xhph3pKndLBtB";
        sout << "d1SecpCR2Op+pbLhn6xegVvm/ThgTZRPqCXrZW/Dff1t5zA2JnSqFqm4e/0Z2DZT/vJdl7+5NX67";
        sout << "Lpf29BMUsHP5HH3IVfIb6AzMa9dSDQlfRYH+7zZOi/0CO8IOrFBnXXj39pbz7pAulSWGCIOlFwSK";
        sout << "RdxE75LCaW8sItXJqem5mnS+avZj4OQ3etFDGUfuyN9rzDWBzhBdm3l7By4CKBRJftya2nFtXWez";
        sout << "n16wfu+ak4eaxZY3F2zEIbyiSbkEURHXDX+TZLJc6NxXsm4AbcL0WsaDqy2OD4Ua5E87IWUQOoAC";
        sout << "3Dw2UfmCovI3SolU0rWs5HFABcCS1j4IEwns5wBCgFEZR2/+w5IDT6S7/uzf9IGf9sk6zphI7mgq";
        sout << "2sZ8sJsX7d3l1/y62kYGK1opu4bZd8/P5FZrW6VZfAJ+lJ748JCn3gasI339OWwm+hLmenTtqgwl";
        sout << "sZpe0tAV6a12v5jla2qQncBDaspVIcl3eHJ3mHaNIwsWcWNQUFApFWyHUYM8TF1q/7pMWepRHRB5";
        sout << "h4FOdQSbrMPOaFDM7F9F1wsN2m6X7Vc1ARJR2RlM1pWBTvw0ZWk0pLJpgo1+BdXJ0XqwFN2MJi7j";
        sout << "BCXYUzB4dGgTEXv4dwINd/yp+40jsOBuIA3l9i/4HvixIxW/TMHOFlFAXnmTpsLqwtlC3WNMh8Oa";
        sout << "mjrs6vjjW9K9w+xhMttrS0YRSnOGAClIqc/naRZzIDnliN8DRcXBWOQf8j9lQcXmFPN9RV7t9fiZ";
        sout << "8exXLhwk2VQ3E/qqw9vXl9emOaYGMvNG4XV1efuM5tpDZ6WDGO6m/DMN4reIZ6QZ875fn4iWGTGF";
        sout << "RLjfw6UMS8b8Sd7NZAHnRkVxJ+qzHhyb2kY0ZLgcxar8kOoPOVkycg1Mx6k2yFS/w4nTE8MiqNGf";
        sout << "zEl4+EtPiirDkHUwEXVB0m4vmSX/G8HS6qnAT0AGPZKKQBJ6ezns1GK2Ig0l4CqOehu5y/7As8Q6";
        sout << "r6Y32m9rs07fss35rjTP1DVuu0DeCsTTonfhNOf4tWRItGVlj6gTMtFlF44k70+LHGz2d/E+yIK0";
        sout << "kWhwOPCGixISZuR/kRpRpxGxtCgQPrA3eh78e90hXi2+tPx6+naPJIa8Eo4EvJ2+HG2AnqisAZJK";
        sout << "2mXr6edXhC6nnwPABi+Z7jBREAZpku2kA51tscBmwYGJMe7R5rpdHj2hVga63Xm1q8NYUYFIKK4/";
        sout << "ce8ocxIWsfVjXPfrskfHLUo/bDwRSehRlQj3Lz4Acd+kUcJjKhq6Km3KXmDQGh+i5zIubRbqaByk";
        sout << "bMm43lXbUevngYxqlCARsUh6RNGnB7kjC6x7efjK8U4accaaMtWJU/ZoOjRGBT96caQe//nhHc2G";
        sout << "UMMhCydLBt5lN7lqg7nNqJOuOEpfcogoFdpY7oHbgOfuoI3z2J1enNSVjmHl1cyN1tI3sBuTtoSv";
        sout << "DhxgHzkqHMLLgaNbZQwoVvx+IEPzA3ZP1hLqyYQWQeCTiSilhdSETP971Cjy9bKd+7FZOcaDY4zk";
        sout << "w+LI+tCOcmRDeYsOBytoD0OnoB5tUxQ/lcbFfjjrJTH3Fd9ukGkhJsD30+HlIO7/q6N6L9hlbZ4h";
        sout << "+fVnM/tDDEPA5K1kdt2uKA63G29XFMyeishlF3ksgTvEWBCtjbVK6dVQctnCUq1mJjVdMt+Gu0sR";
        sout << "bzuZW6Mf0TEhhlgXGgFOXSc8dgBmsOh+XM4QVFS+LGs7xXOfPv8JRg7/8FQWo/VljWvHD7Kz6JQK";
        sout << "eKenBrC3ipoe1LX/ZrViaxm+XTYUDeVYudWH8ZKhyfFevYWPH0yXNUeVgJHtQ7ZkJHqNHuKjJhy9";
        sout << "znMBEKAu4w/efnXsFwYXqygRmvY53oYgeTGo/z4DZ6RBZ3XamRNIo/dde34zN6EGG/z5WJ2iyqC5";
        sout << "2eLMRBG/5zqJzPw23dljkM1IOfNVUrMYJDpqCw1P3Blj8fLmOaZtamjF7CFypEK/8TJO/WaLaunY";
        sout << "QZOQgfdCrP9Ec0EXVs2UKN/2CX3CC8j6Yl97XhssuMdDlgbCPr4beR/f7TKuTCtKdb25QEEEipiw";
        sout << "ZV/vdB5TVsqYMpnCDL3JUBfb3ZNMYQHoEjIZgG+lvgo287uj7YMtomLFg/I73LGTLqunYcWyC7vJ";
        sout << "59pKlWCkn9V8L1RouJLrJGiV08PN9fywa3+3x2nZc8vlB1MAkUrx1uzKOZH6CYyzm92aHjJtPBcw";
        sout << "XCkIEiPqBpQrcrB0aFxmN0vQRv56G8ppY4VHm3H4nrux86y+gMr9s4nhh9IblKkjNWDM9EzCL6/D";
        sout << "6cvJdAsR2LQ7LoDxiy0rxd/BCssPWWcsDMNwXYG1mbJckRIJ6LNsQIBDMics1xs3eLtIbw3U7pu1";
        sout << "9Oo4CSYhXMRy/X1xJYE9wfejOguyquEmpJdEMnRLBn/RbQJMV0JGE0yVILTp+wdzzQ7F+GqNsJPY";
        sout << "gfXxGBpRVBCvIxP+TxgFZhx8aM3i4eAqnf9HqgiZ+jbDZ4l2teW7nQA2v5DBOaO24dIzR84rI1ea";
        sout << "Jcev7qqXkV6P5BjB2Nbf4k/KfHQNO4Bn3KCk+CQv0bGyiIh1uIbt1JbC8FidqYvHzaclpmYBJhXq";
        sout << "l3WRAPIvRjOjsWRHWXB7C61Hv/VABdZEGPmpJERQ2N/Ysiq3mbMR6Udo9taMXtrfXdOlQ2b/RLeQ";
        sout << "vrYjkOoE7o5kHw4jJ09Ba1EIppx5RC7tMKpOCmj22lWSF4P7m403EKNsaFmorxRPIEwN5tBvs15H";
        sout << "A1+r3H2g/Z07vAXuBklqYIVgDxSzoVnakQBWY+2CppbqQdL+2QObiHyHVX03jEej66/TxCKCann2";
        sout << "WNwh0ECClgE1djXBzoOl8viqb4oFqNNvK031gd2YgbvFDGPsF4pwlY5dIsbCD18Y5sa551n50XSE";
        sout << "QB2kbdctE3zn5tgyl6aubzQ6zOmwmgsiitNI8U+WzaizVmDyNzcdTBMgWTlFZ4AxXHkqjQokBhj1";
        sout << "sz0qQaEJxqhGaOWgkANd5knFIDvZItOZ0++AbH7pG90aZbFivfaMtWSa7JIJ5erdabFKX8CiXdPJ";
        sout << "B//j6HVS12jF2iuZMe5Qorza1GLXC/Bo/6TYJOzNDTdy2gV/mccffSx4doZx+DUHjVwTpjdXVinp";
        sout << "/yfost8243CuQN6pjYGOCVoFpu/+YyP6wluZ1IIxof0unhfUnzuAp1zSRZNle53ucVtt0dgcMkEl";
        sout << "nuWt78p+t1pj03kSnkUwxlcZMYZpcjIJtCdmW+d8sS2Fz+3GG7dTFr1P/3HrOxZEX2rbMoBllqFK";
        sout << "wFGui0iN6nmpCJ3wv0PUxqXz5f5oqZLOU3c5jyPQgVBnNwfPyFDAizqDb4l7asbr6HPntfYXAz2T";
        sout << "h6whUXlVG1DYdpD+wASpQnqzg35S0ToaHaMjLzaUayVLFsABV+PrD0UoQmYHEFtevbycTM+GIuHz";
        sout << "gRCsRU7SU73B64yR4B/ZbkmxCetILwUDMr4SQvIRM8PqnZ3S+C94tYJPmq+xih5zgxdZsYZ4jaxu";
        sout << "Bi+u1oa/ml5ffkazFCvj+S6mAA==";
        return sout.str();
    }

    // Returns compressed black hole Q& A dataset.
    // Decompressed format : PAIRED_TEXT(segments separated by "@@", grouped as pairs)
    // Structure : "Question1@@Answer1@@Question2@@Answer2@@..."
    inline std::string get_blackhole_qa_compressed()
    {
        std::ostringstream sout;
        sout << "V1Hxi0JG5qS6Dqn+dZtM3Fyo/32/7lG6IxwYqLW4B3XBEGrWfPBcX3g84coDlfRFCIGQGaPYck7h";
        sout << "p/jyUazTCfxb47qhrQPvvAbyz9FpVNPhEBruPorVGE4B3qJ57+c6BnHTjlF/sBa7aLsI0Rkd3EDi";
        sout << "+HqMFfS6RMXpGUOk1qqkOb5TWuV7D6d1XcxFK+S2spn4s/yW71qAi1UXoQEN4Pew4pyIsa9yhS7v";
        sout << "pVx0lkiGdvdwjQyApyyhC2rC1aHzRvriTYpvwb+02thqPuWQs3YctkGXxp8KlkdZ3414zSjTxpF+";
        sout << "jsUQIyElJILYbF2vW7mZ0W7Sbmt7jQ0+49wKqvGv+m/ThWfenmGTw21TxJmpl37RDl+wcJs61yLx";
        sout << "+39EjVKaulZw06vjtmw1PSSdLHqzB4mwTKxOXnfGmMMh6Yj1ijejr2/zQgaBJiL35YF70fFT1CBw";
        sout << "/jyRGEZU3b6t1QUS6kMhDVbULiBQNyUMnsM59VRyJJ5g4eumCpEldVy3oegUXEQszAQaXly2qcJP";
        sout << "yLRefNy6yq+7+rYv14wFS7a/cvgZwtZHGN4A/4eADYINP35m1GrEiMaxUhB6YCHdGStoh8X+dK8G";
        sout << "KwlLXtU0T2xrJLiVzIpm29QTbDgoy6gIled3xbmLXciKDugw2HDppjgW1UKtQIFPUlq/ttEFCJyU";
        sout << "BqgXe2zfVf9OWBK6KIJ1ycx1dtllTrEDR9G9a8mIxnAhCgyM3INNg5Wo1ezNzF8mMIOPiigH7W6g";
        sout << "RwIl+bUgjcDGaT1LMk/d43gcoEmGyjiunWNv6MhpYjoy7rhi+ZN9B97FWePB0bxjFyUZW8gZh55f";
        sout << "PQ3I2NwM+oJSbhv4k1n97sTHzWB8KPr1y3yKTm0ky/Qn9CYW2qWgcNFA85jyQOaFSzlrlVfSu66V";
        sout << "9p5wYwopJZ1Xg6SKxrgphYipCUBLVCoM+hsbPzZqFeVgeHrnx4PWbw/nD2ni48p4tDFwNR6ctkSm";
        sout << "zx8aJ0UsNTDOPjStiemCm2YozBTAwQv88coBE1zQKY5dLBG4ZnsAIBm13XN68ZVxvLUbk2VmKXhu";
        sout << "yiB2A0Dx+PbHA/Xqi/TGeurnkxBbLz0UWgW5fJnm4OquMpLnh2U8LT09gr4F4wNxv4Xz71XSv3M6";
        sout << "g02t78u0HKxENk2HD2pHvCuRzB0YZt7rtLMZ3OPJaTqSwXVh/bv35/vfQ+SjLyBl78TT5BY4PyAf";
        sout << "0tTTzeAMl7DYGji+PBKmDjoISyVVEsoQPNG4e4z5O6Yb6Cd1tA91ySm5WNoCKh0ySwFKaOvgfoV+";
        sout << "tN1hHqW/gPdxvtSZNFWrLJkWw6TYWaeHF/bISpBLItNS293Tg/6N3/sN8379nUea9budQrmht5vJ";
        sout << "A3rwEAkzUtOdBgJq8rZurw2y6h/0xqIFHOz6uFMkGWKKNf5F6LfFV/fzNetP3cemluaW5L3FuswP";
        sout << "WPCaQzjpW2EzHOa4N9pT120OENQ0cfYJCrb1OBSp5yKJntI39W8rBfcxgGTGQWiN4RsPBgtXcWbv";
        sout << "jA4xhmYDMaBmuPCX74dNsGbakC4pyzBflALxgT7Uw5YkPpVU6h7MROizveeSBDm4MPyn+jd/U88n";
        sout << "kV5+m1LOJPhy7FhJdR9kYX5C3c7pu2S7201KJC33YgTheQ11AJTfsd9ODcBXMgw2XfLS8BEKmHEM";
        sout << "70gyS2w4zJ87c/jGihZAL5KMrRFV064bEwj8g82URMLPJdUa6hSzNK8+JBn//Wv+eh8qjMo71J9n";
        sout << "pBRzv/nsQ2zzLjmdk7YazW81iKW30y7FBuDwS4LIxjybVlFWZt/3w71D378ejV2V4qDdWF+bR/Xp";
        sout << "wC0+IhCsl5tOKB9DVOvt9iSL7xYJ1k7e1Fm2neOk7uf627jw0PAe4RmCptHe7sdznkypxmuA+gaF";
        sout << "+v9snLHOL/r14VxrP75N/Cgde3iiBrlex2s8DHo4P+4dG1F3Q8UaUXt/Wr/3MX7rshGM6TNlldiR";
        sout << "ulIVSNco1t0BLG1Y+zygra29yojS235SW6Nq7C9JdsbeWh0dH+cxTJiZ28OEoSRG8JLGp6liTYzC";
        sout << "R6p641Tt4sCElnoPhW8j8jaQXNCSP/vQKQ4h9NV6REZZzZxfI9VDaKW0vnkNQ/o4jkAmojSirlUf";
        sout << "FoSEeGadEYJ7zFTd6jhAgN5633IybmQbC/erSkUYPHOoteCV/zifxJK5SKQ30+uxdt2SiOFVkNuZ";
        sout << "SROEukDEyb5u7uWx1+86UwZrqCK6Fk/h0Q7xpAytER31+oL/CV9hWcgpg8zsx8RJkZg0mULgPNk9";
        sout << "01JbwK8PgEF2ehy14Z2HoMZuGzMSyimNj54YHCFK0VPDReeDlVAMsdg1uz57zH8+/eUOwAFjnXbc";
        sout << "Dibm7Ui9CCJHm6Yj692Quthvkxb8n+yO/kPXytsO2rynsyVCQWhtM9utJhpCnAPy3edAwvlvDwLe";
        sout << "jnGxj375oYXFqldPVM+spTcbAcC16azQX7rFW+ppHgU60RgCUk0FUYIQmhOiXYZNPA47qCTUw3B+";
        sout << "+47Vf8u2UQbLt6lngaS38qw5Cc0LeDcwSnB+LeoPmyUnCaPeGJFDYDHfbu4qcw8XuVROXwuleS59";
        sout << "Zbr6mvPD1VbzS2XBIuUue0g/IlbCx705ppcDtp3TXF4KRquLO8OCS8WXdnXLe9xO+IgKAoqUP68s";
        sout << "Heh1PwUER3gmFVCHItCMtPD6Bu86XOO2SifShJFcWhb8Ek355HmmZjj1yeg5mMy7TllwRkelgpOu";
        sout << "WXFcTS3O/rVPRzja0nLGMsVSi8igxLgF8vCZMXDq8GO7Q3Z2dNIouhU2Hb6S1s1s2T4YWsdb3NNp";
        sout << "Z1cOJ36T5dNzA5nnmFr9xpSkRBzWFfGFFTQ3RT74mpJpszSbKsl/qdaLMYEtW0l1I+V5fKfsJdZE";
        sout << "Nn/FrOCePcAA1OV3ALyIh141lToZeDqnV6nL0hTI1F7ErP6pKm5VEBS6mbvClEZNVbIolCLkEdoA";
        sout << "Myq3F0whqMhmXk7FipXXARbHMNwTJ4quos2xWENGEUQIlH4knatvf5u2xDlzfAR/be22fHeIO12H";
        sout << "3DO/RvwlRw1sqEKnFqISlmSYLKC32+QNiUucPpYHME4ObArcQXvz/UFY/ycsdZ3BEKYZkYHEVydi";
        sout << "XAkya9jfNhnDLMpyiM0gxr1hk4WVnNbgtSOtLq5W7EfG6ACeacf5Lo1v8Gxr5TacHfGEJT68++Cb";
        sout << "i2SdPLYH+wn84JGFUfykmg/ZA13Q5oGDcLPs9m1fDj6QoC4vLmSWyv+H1dKBAE8WEEDErXfi+QJh";
        sout << "5xQAyby3ER0HYhWJmVmtv5tVEj3c2LqFm3rhv+vilziDpYTMYsTuUizgn/sZ7HCSIQmLV+oiKcpL";
        sout << "wFalvWF1tl/fBo0KT5uF0Skq7dsCTLusOFOwFNv0Lozto5BT/kplRmOs25Lr6MkE/C9YX1NvplZ/";
        sout << "yCO3a0qOAWRgs9taFbgP/Jz7su/yH2CLgg7vP/7ywiaZYJsqUWxw3M1h/R3DNU6cwhScTtCNor3K";
        sout << "n21p4QHD0SYHhOom49iaANkv4/gwYcjeumWyWRmgqIjgRDchcsrFFSKypyRd4YGA7xs5zd2GBBdM";
        sout << "WVgaTG1BW1HMr2ni+AFtJm7R+0n73OutBBbHEixFLv/hXgBqsckLqDd7vk48m/gR2urifnhQT0jI";
        sout << "p6vVC8cGGxZh0pjSP91czRqMM2KWQf//XYh6cDbMU4Zjuo5F4ATKaZjn/Kwu1NXMy8UeqbupAizv";
        sout << "OYu0p2dZgMy8Q8IyoNIQYcrfSWr8hhKhxCTxj99t3FDBsaNWJc54uNW1aKm4Va/GyHa8rW1oL0bg";
        sout << "PbBV+DxA+V/aeOLdfkjiKXVBmP48l8OX54Ky+ELO7HxigGlOvshPBpaRslEslkYrmTMGQih8nRRd";
        sout << "KId1QQYM1G5wOQoSX2Y60nNubF/Etf/hps/Hr1evOJWIDzOD32EeowNSWKEtXLdTyC9l/SAjPEAu";
        sout << "T9G/JQNlHnC+2xHaiHvr3esIOlZ5CoB6G+lLXVDmG4JamPGdank0oqp5yLKQR3Rq2QLFoHaiXeEF";
        sout << "YRFaw1RwENpd9wMJZQHdzbfd88vYsWuOhrNmGGP71SLqbTccbsg1UK9ebQTO5fsK2fsZnBcyiN3D";
        sout << "JRIYmF/oK9xX0pkz+pTYaIi6TIAUtb0G/W+aGBGzNh/cRXVYrwOd3pf/HEIPacwJ6EuUwVo01l4G";
        sout << "zMx1FFjgBa9NpAFYssfpfFG8CagHbPideUwGoP2QhQmmzrNsb34i0QijQOWPXEBrWQdbmoupvVV3";
        sout << "2/lh7g8QfYMH9Q1WUXGz4Wjt28lajQWWRrb0EpyqbC2Y0InaFvZVB0GX3emAcH2XmzPyuyPbZ2E4";
        sout << "aUIaZI0GTrW7hWnCrIw5/31lL5aI4LJwTJox59JUl8pXiwamx7CX9m33Ygo6GMM4wGwCzDX4cCKh";
        sout << "yrGG6BvMuJFq7iYmKEajfjrPAxZKcm8cFPGB2mgW9Mn7PujcGEFzCVjXatArm5w5nIis4WstZoh9";
        sout << "qf6kLRayAM27loFS5uth8o7jRx4ZUNJwUdOCDe7AnIcE59Dv/dOB2HaSr/htOgh8VLiphZl6DApa";
        sout << "wC/B2kyv7nyG6hHEP00xvD8cZJTcl3q/IvXeVQuUPKzUd1fTsl6dgMqAACuVfA78VdcVuBqhdWTI";
        sout << "3U8awa900zUUMGDIvUrRq1O8o6wGzqxkZNkXso2ZBbN9Ftay7VwIC8EOZg6a6WV7CVCLReHbzn0V";
        sout << "AMdSFz9/E0nqSL6SqANLpVJ0i/k055rc9LSl4MKUVv8soe/NE67o+oZpNpyRS6hQQctHfqOW0yPF";
        sout << "8rDiMKCNQifJvINY6bD6Qvm6p+if/g796Wn9W91whOLURKvPmIxcEvENg0mPTKxwvZhnRuZeWO6e";
        sout << "TRM/6L6NNOFaAyjn/J4+UMlX9+Vaaljos6W+9CwBlbpuDUV0L+Q+yMkWtrE3XCxeiooQfbw9yqXg";
        sout << "iCoUvr6JvglGLtaPki8EqDIJ1h3DldZ520gL5R3T5XChuEyGiSKddJyvNVFQ9qRn8SkD0+DRYGnJ";
        sout << "sj8gpf6X13TMyjeAwcp5ZHoRvjkkXJZBE7t2XbCEA+XkA8NRKnZiIdEtDiIibNDLn+sgRnUMakod";
        sout << "Q8EV4XVPDqaWgTirFjkPtZ0GoX/c+k4guZxFlNHJ8+fbf6PqacJODNcOLCQjUTshNbz4drq/eUHT";
        sout << "cqacn4Vxk7NMAQyz64ygOknmmm5KDQErO/8n2eUenoXOhHJIx3cogxDapZEVDptgOTNLuTjZhyJl";
        sout << "2lTMfSRusXepIAf0r/TGkAw7BmiNeH5qiTNGvXvcpB+4XGM6zl57xwaGIfBVYDEt5jr6TZlrqjwO";
        sout << "0Z6HMgf20AIBbUwDJXh5696yPphhAUJJa+vodLUiaVloSKd1gIPlJBcdKHex2W1XRZEMZdSNrW2d";
        sout << "B3IJ0paMvjswxM9dfftlls6mv0v84BIqNRFsCdMM8vg0RGAqnkdml0kV6dLGrffk0i1K56qiKerc";
        sout << "c3VptxN9kqeWU0pp3DhgJXxh6I5L0E1BSKdV6zYuGbQtTwDvEz9niB/vuB6xXuh8Y4uJshnHGhjF";
        sout << "jJU0HfN+5XRAoM+5w+Y2LZVmtC2n3OZh5dK8f1JvVvR5Ql3RFptj/4+N906g6HUCUXJyrQZFVE51";
        sout << "tbraKqa0JwT/23DbXiFlcw8C9xuKYi+irTpEgeUZlcX6LER6HKR5QN8+br+Ke7ph8gaS7BENRSCK";
        sout << "WiS2FrjrAymfFziB9Hu6jlnYWRrMAckRRMGOYaFhkD+jCn3U+jg/oHbp8a2KFMkioV6chktC6R1D";
        sout << "2n9JeCdgbqW5yy2UV8x0VV8XZu4Sh+flIRJGA8RSEGazNOXXZ61Ijn4IQXufEKtXd65Jt2FucEI7";
        sout << "WsG5HavJRoZvmmRPBU8nL+SumUm07NstA1G6TS7CqFg47+msLiuW5f0tMtfmMTzCkf6hABg7tTqa";
        sout << "WEDwh5ATbnjzE4gPyEx1XUEyHmE0tKIOurxee914hkThy7Bkh8GQmJhubmdTB5t62CPfJUjDEBBD";
        sout << "DiTeW3ycHZ45uJmUwxfVjHQufLioZq+1SraZy7vmVW6wcBAnoeoAUa0VxNSNQRSU3rYK0HzxJcXy";
        sout << "NCnHHt7QOeENJHcvKGlw+SYdpVW9/Yr9xgu4eDJnPG6F8mS74SeTwacv7mC2r2R1YSTtGAq/NlvF";
        sout << "HAnRL+B6rcHkbMAQlgR8Id6gMXPtM8Y9HQVurZuHfs48o7Otkv/EOlXhrSRfrPPagICBVvCIX5Hm";
        sout << "d/WVH2iRXqFYLk4bzybtTKU5oAeLmqidXvfXTlhbDgksY+ueehXi0TWEptzk3t0dfQY3Hz4j0aV5";
        sout << "hiD7i6QKEC5qoT2VZNIOnjLc1iohYUipAl+I5gh9Vfdt9N5NsEfpwExNrNhJXooZarcThfhFfjh7";
        sout << "2ojq6BEKn7bdFYulgoI3Is8oEee2k8C0HLKbxWAijwIkNA534QKUqzc0zT5VXS0/XFwWK0Va53sp";
        sout << "mrh/vGH7M/l2+mlCuJEnhbES4mWhM1zF+WlJGuxg2j/5lThcDEG1yQE9c4sIWvAPG2DocoYOb9AC";
        sout << "VteHT2if3yMc/zcbV4wy0PkYxgekBMyGHDscLEXCF3xJXt4YcZ4sr2kOOyMW22F2HgqcYkJVY2ky";
        sout << "JNydERjjeRYesrJv+ge5xOqFE/5hgOZU7x9S16z4Y8uAYef6xTkRLiBe9iovekynKCH5Iwt/92Q4";
        sout << "DuR6ZUvFrPWNkzZb3MPt/So8DbsTrtUQ0jhRhSNgppJsrd1yV6jsNA9lOIBQFepZ3fWkUmHq69Kz";
        sout << "k2PCJ2zVIv50eWmLcmgX/AnPIjqfxfN+B+7aTCL1gAklA9FRzfe7yp4PyYo6bLkKwdgU9IF+8ePt";
        sout << "/qpDmaSH3MWvZJa1H4roF6eYfZ9CNqArj9trA9Uw4jc3IyY/esEUxQVCdLGAMV6cKttsgXrMzUV0";
        sout << "wKwZ6eS9zm1E/oKYoGtIRMf/838QKNOFY/XNRCGzZzGcPG0gxZgqhLRPE3EUbkaZJSdDYsMPzkAa";
        sout << "6ANcmxXCczvpf3sGYLl0z01QZSoqmGPXBh03A9iFPGZhEBDEFfqLjCdjboAi1JBqRoUMtJCljsAS";
        sout << "116oVUJxaC33kCvYJDPItY7O78aPOKMXPvxSWt0yOTm1qPTuD+tZcpPIdBQoqnC29m1NhcU5K0qt";
        sout << "EbusYN7Ue612B2OSzbDRpdd7YewRqWQuF8iXAHE0eQnoEBzL9l3X6cx0VE22OSlGlmJEjWM5IaOB";
        sout << "B7DYbK8cWdy+/qAtCBYIL2c8Voay5zHWx5oREab/q9EfHa7yn0ZkXjsSemc+0YJIqZXpHPwbNIs5";
        sout << "GB9y5rzl6ZvReHeLQPs90F6pjr/GkZQtvTeu4cdObCR+bsLWP6n9tOSkmMnutfQ+5Kg/RIdun+rH";
        sout << "Uwh4bCiknswZO1DydfsZQRkkbHeVC+/1AbZOP6LnHPAvoL8nanUqJfs6MLqzvgmrWIvaHnPMUPah";
        sout << "e2ST5ODrOU4eB84XzWabKC8CXLzAO+FM2YcPdRL9nlWB4IQUKOJ05sGp3qGFObwMO6b7U2q4ghoG";
        sout << "DzJ8rRV+2StFHNCDFVDUemTT6BqmBONHXFs7ynXJ4bzR8dQfrJf/DqqJ18bWS1fVNfW8MrNujk6n";
        sout << "SM3fLzZ3iFMR3AryoI72iKyuYPTkh3/V+jTkyePN1mSPFAttjOtK4LHadFbY+7wtsmB9TteKpqXc";
        sout << "/FZXur0IJ0iA/6ggxeNGKXfI/oBBafS6ZiLDB1Dfd16XjwJYoqV5Wic5qhW744zcYaxH64MlgXRJ";
        sout << "0AVkbl1hHr9pXtDYNu/xUID9zMJwHUikYTpDIcqa9YKf0yAt0L/lg04iO4mrF7Anx6TDkaqklLrz";
        sout << "PlhOGZeIa3ZOMU73BfDhCA1js1hpAZvRMuMP/XTW/Z60jmHpV//ypwzNISgx5chzlMhP+SGV0rnK";
        sout << "+MOCzT1Lvbji6GVn0hJHOXmlaEKIphEA9TQ8hWJ62J0878HwmDDzjg9oJfuiKTKrqE2dU/xawHZ9";
        sout << "D0gA8HJLlAmzvN4BiMWjLl8TZbkgKm8UH4+VgMQsXVuiw6QR1u9lPw8xUwOXvnYdFk6LbeJUj84N";
        sout << "J6KNBDe3vvf7AIASxKskhisNipMNk1fDppjTVZzMeUJbuD6gowg9yiG5O0MJKHpm608uvp0bT8CJ";
        sout << "BmrMrZYvdsv5l1H1N6pKsKA7ig6nr9u+jhnhKEl+y0fGzrpFOMVI+IUbxh2yr1HEkpbQzznNg2Jz";
        sout << "giH1vz05UdZuNpoYSaMZyBjLocXntc6kMszrX56ygZJSQf36bqXjDOG+NqcEJLH7LiANA5ozEZje";
        sout << "GwXu869UeGmD+XFn77l2jOKzycPWyzNIfEFhLL0eaoZWuCloD1VOP3puwcuqHq7zK2jTHWGLce/C";
        sout << "TnP/8vVCa7mgIx4wd064Kl1NpWdnm4notqw+3CkJUXsEuhSoch/vY2Op8g6rGUcI0D2ijMglAIOa";
        sout << "W73CwzXXZ86OtI6/tBmXCFSy1cnW3HwkQIHsOQahHz7Lbuy4sP1sF2571mAv/P8CvVvLkcjSpHO2";
        sout << "TlmCYMMZwID4IY/eYfTZMgBRLnVZNhy0PyYFzM31oznv6FH6opHFyXB/DopSa8m74BxrBuoJOs/b";
        sout << "IRisSYl0laYhr6b7OiHraS5udN7dBzhp8VcaXvHeakb3LqdftxcV+J3bEhp/m5AwjmkbfMLau6Di";
        sout << "FIfVQgHagxgvvC1Fz9Tryq325uOCWvJGm9zkJAfhR1m8hWqQKr5GogbPO3H8HYSK/iAPMJnrOgSo";
        sout << "bmHfyYuMUZ8vQrQsBl5Yj8W8pRa1MuvFSezrg5xwJsDSVT43XpTV3+3WH9s6HrjoUMvAEvPOuQQB";
        sout << "wAqPgjiufL50Q9fw689gt1Z+fEvs05yt/xiHTxRuJRM2ocKTwiRHPzjR6z88oX4gRybQ12eKBrUv";
        sout << "dbR7W4qZs25bAdbb5xkhjhTJS1PDhmcPYNOo+6HnfWoAN1BNo9kTw72CQ0jvD2TwSEpLkur3v1Xm";
        sout << "hmCWx+cvPp/Nt79MdhtVZwSlYbsoTYcks+zSvmZeciOhOGObXh+BKjUHQyoMaqrU6RE91kzrLkYQ";
        sout << "AEGY2Fzz7pkJErLL65aATz/y7unKHGBJNjxpTX3IpwHq1br9q0f2bJK9OeZ9qn0VcPPDoUelQaqC";
        sout << "LXZlop/p6eBKO7LienYg0bSMdrLBcXjsMaOp2qtWNLy/z+SXGlXKiOnAk56Jil2C+JXuOjfLvcxh";
        sout << "YVPGUBLEthhKwLV2S5ghjLQLdltcJja2WV898ywXr7UZ0dDCZKOlgflCbFex0eXkg9/ZEByThCyu";
        sout << "wHRMdDmGIZeCV0N0zDU+Z5cphZ76hCi3Oulg9dydcKsmHj0Lih1gH2fjLR+k3Ff+1OQmebC4yAiG";
        sout << "KZ8r6JxeDnQPwHp6JsCsCCGX8EacDTyPqslUgkAXiXyuP0MdZ6PZQs75oP8ARILoaqqt7mfwmvJb";
        sout << "KaA0fVweXO/y6ApNRklQxPLJLgRaQscieNbpYZAIyK/UTKAVAWS10XG4J3wSciA1+djxoPb4lxTT";
        sout << "PkuKmGbAhAoJnWevC+NjbuyHOHE0oAOwyUIaMpjmI0rWVzHZqCa6PMQ3wLAq+FhjLQ0AvDBJpm/D";
        sout << "bQjiLgDizuMvS2S5OqJs0JX7S41gISU96LwL9ZP51k1sOVn0eosxEgp5Lvq9HHDiNpSkpCgTojN6";
        sout << "lj+0mtHrrz5GHeHlSZcnkH13ViVDkq6gMnclAjBVzJhyG1MPsLzzI2dKHAooicFr0NPqqdLjJ5eb";
        sout << "762euf1DHOxAw2tuwMTUJYsFkPmwT4Ot1yzORawWmDtpdLT6yLRdtIrJyNS4FzT/KJbm2Olna/Av";
        sout << "GS8cYWpxmohT68n0G54zGsFc2MGc6kjs/jceF4PQ2YVNyEFmGI4ful3FeLllmiQkg7eOTfYOrtYb";
        sout << "pAJICCnbOnILeAkUllhGYQYjAonFLsJlLslIteFCsf4CjDqai0qN1rqFk6Tx5gCwY6zzHZd6R3/B";
        sout << "458H7z+pvHpcbl+7/9NjlRIY6EtzF0AytUiPzPDMrLbNZLSDycMJAzZEZ8DWqMnmLydL18fL1L2O";
        sout << "INvHWD/QNe7fo5ugzJ+9xSkjO2hUsSf7Gh6I3dSkOIi/oFiCNIiuAeOVygD62wA=";
        return sout.str();
    }
}

// --- Generic dataset access interface ---

// Returns raw decompressed string for specified dataset
// This is the base function that all typed accessors use internally
inline std::string get_dataset_raw(dataset_id id)
{
    switch (id)
    {
    case dataset_id::SHAKESPEARE_EXTRACT:
        return detail::decompress_data(datasets::get_shakespeare_compressed());

    case dataset_id::SHAKESPEARE_PROMPT:
        return detail::decompress_data(datasets::get_shakespeare_prompt_compressed());

    case dataset_id::INTERNAL_TRAINING:
        return detail::decompress_data(datasets::get_internal_compressed());

    case dataset_id::BLACK_HOLE_QA:
        return detail::decompress_data(datasets::get_blackhole_qa_compressed());

    case dataset_id::PHYSICS_PARAGRAPHS:
        return detail::decompress_data(datasets::get_physics_paragraphs_compressed());

    default:
        throw std::invalid_argument("Unknown dataset_id");
    }
}

/*!
    Returns dataset as plain text string (RAW_TEXT format).
    Use for datasets that contain continuous text without special structure.

    Example:
        auto text = get_dataset_as_text(dataset_id::SHAKESPEARE_EXTRACT);
!*/
inline std::string get_dataset_as_text(dataset_id id)
{
    return get_dataset_raw(id);
}

/*!
    Returns dataset as vector of text segments (DELIMITED_TEXT format).
    Splits the decompressed text by "@@" delimiter.

    Example:
        auto paragraphs = get_dataset_as_segments(dataset_id::PHYSICS_PARAGRAPHS);
        for (const auto& para : paragraphs) {
            // Process each paragraph independently
        }
!*/
inline std::vector<std::string> get_dataset_as_segments(dataset_id id)
{
    return detail::split_by_delimiter(get_dataset_raw(id));
}

/*!
    Returns dataset as vector of string pairs (PAIRED_TEXT format).
    Splits by "@@" and groups consecutive segments into pairs.
    Requires: Decompressed data must have even number of segments

    Example:
        auto qa_pairs = get_dataset_as_pairs(dataset_id::BLACK_HOLE_QA);
        for (const auto& [question, answer] : qa_pairs) {
            // Process question-answer pairs
        }
!*/
inline std::vector<std::pair<std::string, std::string>>
    get_dataset_as_pairs(dataset_id id)
{
    return detail::parse_pairs(get_dataset_raw(id));
}

#endif // SLM_DATA_H
