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
    BLACK_HOLE_ARTICLE,     // Black hole physics comprehensive article (RAW_TEXT format)
    PHYSICS_PARAGRAPHS,     // Physics text segments (DELIMITED_TEXT format)
    BLACK_HOLE_QA_PARTA,    // Question-answer pairs on black holes (PAIRED_TEXT format)
    BLACK_HOLE_QA_PARTB,
    BLACK_HOLE_QA_PARTC
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
    // Returns compressed Shakespeare extract data
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

    // Returns Shakespeare text formatted as a training prompt
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

    // Returns compressed black hole article dataset
    // Decompressed format : RAW_TEXT(plain continuous text)
    inline std::string get_blackhole_article_compressed()
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

    // Returns compressed physics paragraphs dataset
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

    // Returns compressed black hole Q & A dataset
    // Decompressed format : PAIRED_TEXT(segments separated by "@@", grouped as pairs)
    // Structure : "Question1@@Answer1@@Question2@@Answer2@@..."
    inline std::string get_blackhole_qa_pa_compressed()
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
    inline std::string get_blackhole_qa_pb_compressed()
    {
        std::ostringstream sout;
        sout << "STcT9t6YD5yY0kb6jxFqk1otvRFGNz06Gmm/urUcEc/hmL26So1K1PPLssFdh6UTfTEDE8cI8YBU";
        sout << "D1NV5T6iPY8ebZ7Udn083XbB53S8y4ViuruHTHbvfeZrL95bG3faPc0UWkxsoCjxGJJ8KKCxukir";
        sout << "IJrTYGeoGunvhMItC2bvN4/af6z0nnX2DzwpYf1w9rPW9Exj42O262ivlVT8+nsPSFBxqM45ppPp";
        sout << "mSFaoCKO/0EET/GKnvIW/pLyrnocAgwNYJWJUmzNIhZf3uLHMisT/6izIqSZmXdZVr0yfvfplAjO";
        sout << "F/4J6VQ9rgl/VQSwEryZPqxkWNPNHls+gCMWqqVCfd73x68QIBaPz/VLZTBoFiiv4y2Yx+VrLZQB";
        sout << "zxPSuo/DNQjnLzGrxpju/s7se8qAUEZHGHSc4pp3QwFGudJiZuIcvY2LftGavZuZl9WevGkj1hGN";
        sout << "hXvkpRW7tTk6qHD+97Di7zPG0CiZ/iEu4WwDxFt+UE50yyJgRGURrUiBhWXyXBZlAKRHzOlQr3B+";
        sout << "cdXrqQFiAdqm2mXkRwNL6HlOSvvBq1LI8sO7amM2XyAFHbzl7Ijs7M+LFV9dbTEebKdok7ON8vby";
        sout << "iV2X+Doc5SqXJ0MHCvKgvHfXEhTld7zTv0tYeGNxKG3VjpdWNrJbd9ni+6EXm0NTnMJjZE6b6Nkj";
        sout << "iAOdRY0rJjC4nqr70vhqhuqa9YmbUYc+cF2CD1tM/wiIE9f0Ohh0rcgmCG+2LcTDQAEArJ3OXkDz";
        sout << "9RVF21HGF3QGKfLaTcFnIfVRzxWZieDZJUkiT4dqL0/aD0Vysx3ClHzqc9GUW1MyTOLrbfsYUq2Q";
        sout << "yAyC5CBIehYVZf7t7VhgsQwqrzPP/TBuXeICcVrtkWEjDAF8EVFq9s+6FKsQHyNb9dIkBpgQnFrv";
        sout << "KmLXDUYr38fZqaZs/gECGcpuConuN0/pleNzQqHj0Sf4WYK28SUrcH/ZdSp6dv7YH5ur6ATggfy4";
        sout << "JznD6en6qj8FVKCAt/Mau434CF7KMdrMPiJ2uFccYIPvftrJ1m/44xNSILeu82c3DG9qe/s4IVwn";
        sout << "Nc73HEPij/XuI+V8g3KcVNOMR1BNmZGpP81KK11O1cnIHddZ38MI2Dd+xeHQzuSDM0oHnCnvvTZo";
        sout << "LJ/0angQodJfNaWO36mNKl/a1nLQpvfm1eO5CjX+QzptDfcmLdB1xWABKblml0EyoGki7I4zpm4Z";
        sout << "AIW+jriY7PjU3KkIcAI4NPvjg5fNYQ6hQP30uMgZ75M6ujWG/bvzZhmjeEc2IhK/1e8BzYyfI+fb";
        sout << "/0Wl933WxSy9UrJQrusxhpLWWrpQdVX2J8mCzBH70C7A3WtofH+mDh11xX373iFGMfcBDELv149j";
        sout << "z8sI+VAqsxnZ41RaqKX5TW9i5k2HvA+jLzdiO9OZPeor9mC2iOdzRApuY0gHPNOYkF2FS5A1+uUo";
        sout << "LIxPXY68kf4OW5AVa3nEWYP1seFS2yZTcSFNpNpafpM7gtYTVLTbf8/U897O4KBiVL306vAOpFWq";
        sout << "1zQqQce5aI1gTeVkW3mBdkgZCULZe+o726L2X47Tk+qfW2vOO9OyXlX5e4ZCI/GW7z6yDYFO5OPl";
        sout << "Sq09hvVvO1ZiWmGLAP5tn/dgQnnCQ86doc+W+Dw/JsporIxcorkQ6feqd6cxgBTZY+8yl2i5g70X";
        sout << "Og8SEy5f4GGsf6ahxLZygRfeFQPbWrETWRIhwpZIKJUh7YATU9zWQFRVg/bPUMSBv+vmbCdQKwjl";
        sout << "tMvHraMOyQ/4FgiGvB2Lf+n14fGzkt/UJ6M+J6yXZ8fjDz7OQwzZqy7xMbeWnS8LyqJvh7j6spW6";
        sout << "zNRkV2eFe+6Qt017mpP8zu71KEG6gxtGc6LxgsMp7zFpbr0AXCLZzglhTnl6KhOcWuq5o5aiwm3C";
        sout << "0Wq6q2wSfx4I1pQqF9zwM4y0woQ1s0lew0XjEIyoakzY1bCqm/IWBevlBKMzoUIv5uWkj7dxGcb3";
        sout << "Xick6ToU8+XKP18VPfTZRQw+wRg5mMP7FiPkBfBVKgKwSfAoPr3QCdcrZ5N6Q9N39hFlkogGfNc9";
        sout << "Gwt/ZLfqxhqeUTSCXQb51engp/u9sncNpQ4ZAEVzM2NKYicowmuYrFouXd7kqcjyJ5fRtg61MkWr";
        sout << "f/Nw3aoWqHjvFXR1eAQBNNhS9hL/RQ6lldO2d4m0dgY4qKaKNDw70cXlt3KX/UTpbnNT19oUYaoL";
        sout << "d7AKjX7Wg9WDJUDCynduhXyVwk1DwhX6Ir0MknRxNumvAofvi53Tq8t3Sw8205yiIontth5MhyaD";
        sout << "RP5oEtueugTo2RPqsO+CFAchA3S3SjvFiq4S6O+DmmGwO7Gad1QFS3uLlKSXmszePwtcWS1OKZru";
        sout << "ZpP1w8qtWVLn9Id0rremLEtKqymNsMn0RDMO/be0IL0enLwbefWOBvCI+6GjEo8IEf3VeNMljcw0";
        sout << "uyAegqDKC5t7ZXOTFKkrkF0H09g5eBPluawGniy832X9uELxFQ+04MHi6gKLktzqtIsH16OrmlOk";
        sout << "0Vrxipar/oGMB6MsHK+ux4HGTeB5ab4iOE2exN5wHL94S/MMXuiKLOOnY7J5mWI0BARg+0AgOVIg";
        sout << "5b0z9xaA/K+ZLlcr8ggytnOymR/1jvo8+N88N3f8URdaDU7OYEnrzDV1xXvEysi/lTwM/mINB//r";
        sout << "8IMxIgraGF8zZ1eFEbNqLvU6BSSsUWAxndbBgvEljD/5A4eIHAOTu91BF9oGjaH2gu08St+QPuNN";
        sout << "Y20sBSa1+shl9Z2BS7ZS4XUZHjTG8a6PvuNYzLD0/d4Zuiemm7ifH8qo9QZx+J33OIvRqla8F4Xg";
        sout << "W7VuDtJ6k3WnGylUZcouHtvJx5RUfaGE1z6TAKXqdO+vYx7eGspcG5opWKtxFhauxtK/hhO8xfLj";
        sout << "ZGItG1KQFZeTN9Zsh+o38ZhTCayNGF+ylar7pVEvOjiD2ct6iVBsdJWtJQDzFHATCwybdK2SZVVy";
        sout << "5uH7I+//Op2jFMHvsJpgXQRkC8mJZEjF2bVze2jJiMP0R0AMWAP3H4I3K0Cw+VBHOUmKJbdVT2dU";
        sout << "/Aw8tI/t0KKIdrGswmiOeQq5zaKUbJhjBspHi2mpSGzlZ8xOUhZ4IdZDgfU+CfJwlXwTptBK/6Xj";
        sout << "POdcouLUwvtGndMnE5qOXOwycMnzdBFlvgZFGAEsrpoKlvsYNASuGtTJssRaJz87+1t3qhVv4oaQ";
        sout << "t2tOnZ4l//EMI5HysLQGOy2c1S+ujWo3SL3uC9osOqbzEpxCLuAvj8Thqg7q7malxlF0SWacwA8/";
        sout << "XTYCcNNpsSGrHN8qUymmGe5ddK7u824jGctIp3JS4eqGoPAhaozs1ffvCUyhO3Va1M9atbQ1L/Dg";
        sout << "cIKXLStcXDuBZI31r3IfbBit3hYDX+DwDzKLK7NYdb9pCoMWljG9hciZRawmB3ed/B9V95AK74Bk";
        sout << "mDtUCmxVAtTPFrUEqdqLrSJTXoZ2VCmEVxlH+NQH1dWGK1Jnvtcp8HEh6Zj7I+Nwn/1UEOlFrK0t";
        sout << "shXtlpioX5gO7bD4QZ9+evj74asM/Ynj7P7vYmsMpK8aXNbwiGFEKpnR56K2Z28jTR+rvxKokzDd";
        sout << "tirCqZzK+YGTdVfindxzWUsjtXcZcgceHMmOoF5oI8WFyVm/NzqPElCpjcgSwBOXjVA5O6xGBgHi";
        sout << "8sv78BIomBfVSLCCr9e/DhHtJXDFKWcaJVGGSBnGP7X+OX7AKlCoWCpudwDbNM402f+Ypf+vULj1";
        sout << "mZk9CH7bkIKMqBT7UTvUcJOO/fuViknn0uqWV7ialxp3P2iSOmAPn5eN6wH4Ymna2M2usB7fph/m";
        sout << "C30qMFwzwBJhyJ1plbeUgOwDw6oVhTXAwJC+8PasaKfpjwqZZJTgqpBV+X4ROqCp1nRNa5lov8xh";
        sout << "ksG3JvignF09Q4HxkLB4U5w3bssQyhjTA92NXCHjYc1jw2U4H7b6XPMoyWth84pTcJib2St8qoVj";
        sout << "zQojKuAZl61qhro2E20MSsaZmyb48At+xVbryfy5H9GXM0MMxxQWPb2KkE6NPZsPS2W053jo4bWL";
        sout << "MfdW4UAcLRuQJ3K5HhXs7k0LhoJA9VR5ySmmK5GcZOXIh+Zy1dqH02MOtc/O7v5zw6RuyJS+ob8L";
        sout << "kPeAkZHqqdlnpVOP9vkqJCZvuYDak21F+yHuZ0qL7SzFrclZPYaT71HxddTkolOCQTachpKMuub9";
        sout << "v3JNrfRxh7diU9TG1cD3/Cfq4bCZeiUbORDp1qnLZ9Rqyj5e7vJVsa/Sck/pof00NsGETpAzmY9W";
        sout << "a0KcRIOrZ/ZPP/KswEcqn/xiS1MqUtwnLXygvcBH6WcgbYkgIDYHLeTDgI2lDwjWmr8d5ABbnkfS";
        sout << "IQAyKRbwolWPAWG5IKD2qYIAwZg+p0Mf+IMlGndh22Q4cuSvfMsV3uAWIIojHphX+FfHxaVxdHIO";
        sout << "PP4hY+Nk0uDTkfncG/Z8L0WLk59UqfadpuerxkyAZx7nWhqiLyGcLbdTif0eJ/Mzl06skSGWBHbO";
        sout << "ISM3olIJkMNSHbP23Vze22hcHyfOm2WfJ7hW+9YMUYdeQl3yLds8n6rk+HhVT6Ase54ZVRVkAWzW";
        sout << "lIfMB53BIXf4d9LK8wjhgvCKJh81su+h7rE6RWRQ4uhjj5CvDGx1w/V1AR/VgaCfqNM7FhJI4+ZI";
        sout << "3ydy7gvv8Hy6jX1QX5vFntFCLTG/2iNaVDVNy3gpa0lTbwLyB6NRT+G81RLUo6+NqF3bTIm7JEnn";
        sout << "0i/wNj5iOzGRzYPE4ot/nDw5PTdZkbNwfpCwvBglscUaNKSGq9U1jqnIHPFwf05cFW9CG5mbM/iD";
        sout << "0TuEmXj8J31nDLSqY8k1G9xA4Cmg7j8rgT53UFrusaOHv13MVYINll6NLNbCh6mqe+kkG2nBfxPh";
        sout << "jj4hgoNgPWcaYakzfCopprte0MEbpnd5MbbzfinDEBtZZAZqJqw4jesUh5IwE3Lbl6pT6vy+P0Wc";
        sout << "BeCW9Esw25A95xt9KWZY+o5xTL/znZalwdAg2E72NPh+K7mlaB+UyoCQCqoyGzCwfzSf/hCt8XUR";
        sout << "qd9EiOpdknOzCRzBasC4OtVQhpkecDa0NI6pOYPjZQHSlPrLQyDKdy9EL62Y8jtH4faDm6sd5pKG";
        sout << "6PsPD8Khg8zfMw9a6rgWPImfLb5SuW5Vt4MHP8s1KCmKAl72oewdOURrVWYHyfJ3hQpOyjrjnKIJ";
        sout << "q9gajL7FCPvZXK0ysKiUdEpzJMBWVUhT2dalVuJ9Twj0RWukdCKsgICddDic6oFZUsJ/JSQxmd4i";
        sout << "Ua62IPga+6AySPx5xem08vg5xVcwhDiKCcXIJqetldS51M/X6aryylF+dBHfExky1GOI9ceLV/QO";
        sout << "8DJ2AOWrAmaNiemkPW78DBowJFa9EfZ0uLbXyzQrypQeFx0krHfWJeF1bH+t1g5R9aedXRooBDtO";
        sout << "L/3DWgemswt7Q2JvRhLk93EktsFgIFbKwFqSDFkvahZzQ07hY3jLbW0lrfR5xtZEp/SMKTroxgml";
        sout << "raCftEGq9i0+wzSX8ANJj2JfolnlZmA+dSgMwNNB/Em2XCyUYL6XpKhNNXrdORF/ityrzX73Rnwh";
        sout << "YzivtT7pBy2SWDxiFH+1JVi4mXXORDijEdRa/dzb1TVCqoAizA+MYcn6U/2TgnQkx8WSlxP7Q7pJ";
        sout << "HizIf9yJtl66ORlbIKbQu6DIRa6Xw8sZYvwOiF28RqzHJhZfP4fL09/wq7lXDNAW5TaK8VyjmnBJ";
        sout << "VpRxajKQpVVyDuwDNULTwt2jBqGw3jbvcjY680WzVJBEC7ItTRreXrhKEUMXyP8rzlDNZJrMZGmm";
        sout << "zYmZCv2nGoAJgVScSOHyHW7lkiva4jz3FDkBdyztvcvuwj/pLX8KHbWBxsKEUjWNEo0jpmvz05Na";
        sout << "mCS4QC1xWtsQSQBd65uNe82MeJYqxcdMfI8vSjcrvhawzYd1vF2Rouqcb6qJWfL8ASiA5c2EgRh+";
        sout << "Q6MhlW9SoKcGVc9Nk/WtWFBnPDQwFnsZGyVXK8L+K6XeB5r3Kmr3JK8kFbM6FNh2cdmOOhcREs88";
        sout << "lbjYCdaijEm30JhW6uDMLue9Zh++aelebwIqLqFu24FmIN9HZvaBVW5K7U3T4jTGKiW4Mc42nKti";
        sout << "qar5A17qbGZfY7MIQtzrdhNw7WVafQ8Of0NzdkTK7e3L08iNTxTWbi9jULa3t5eunBPreiWRSNhs";
        sout << "RMzqd1fMo0B0BOeDqsgEFBR5f5vXNmLQQggrZNqkgpGUq3pafy1ub/OjUCtMsgjSvQkdQj+4kCeX";
        sout << "uV5zpoHziGHKaSDbpwFQinn21YLSQ1l4oJ9UXR8+PTtN3AHbD8S4Y/0Wh2cR8OE4iswnHELUDpdI";
        sout << "GFQpV8ZB+hCOFZmRzWH8X60IBp9Hzwiq2ov/EoOyCMIOCFm1Kzj8z6E6Mjq05oSGsx688Ov7yL+D";
        sout << "3nbN6fSPT9ToZlm9XHbkzYTAVCeda5mVS+hoo07jGYl9pMmCxye1OUZ5iJAz+oxJOOLR6LmjYmw6";
        sout << "aWqtVac7LkiVSNCGF6CU5aMYR8jNsUtliRfjAglvixJ4ugjR4ZCly7PE/Ut/4YJVfpn+puZcsjYY";
        sout << "EXRFLJkQGhphdBNUluJYMoz1hJVQa7AayCApU22Q2HTCJRth653aXlnR0R44ScHm+g916G7/rMiJ";
        sout << "puZ6dET2p04/7B650kDJvYOmo3/AQhZi4ZDB7m8W3KZz2f/YiMJZK7lt9hBeZgKBtvJeZ1CRv/6u";
        sout << "8+NrL/6c+yux+XdF5defFKmjebWsnjONEu0Ry/+ntL0UHnQZfMRaXNZ2GAvTdYY7aGdNJI7IqWaQ";
        sout << "XG+Wv3jwYGjIqBGzFtK0cffKl+czAeEpFvFDqij43T/iK0Kj6MqeEZkssrQ6zOQtcgXAv4DSZdyV";
        sout << "oCuGZUICl5ey1E6apsIXg3/TenoRdMBTrCDeOc7N7CkUL2/mKyicst5a7qKNGwtlonESYl8Csyvh";
        sout << "IUggs60F0IDrlp9rQjmIRKc7YLsB+oXaTgmmv2jzs9OVZW6qo6eKCkMQbl3eMAtn3saoe8qMPmMv";
        sout << "NI4hZhP2G4/TlaWPTZqVlFTtd4yY3yCYbiwUtcVAzBaoTPJWTCH8PcOOzqBpFVrRsl3kjekWWHIC";
        sout << "xGeJzXwzAqb7EsPfFzGPrOGQlK26LNMOL9LV3MPXeZcGRev/Tilrkt/y1vr2WnafJgq28nRt+NUZ";
        sout << "HLL47TAkYY+lf1MibIUkfJye29aUOw4vrzk0dbdmh/FrBjZCEf/oTrcjQ64YAV0SfMft/AN7l6Fk";
        sout << "aZBxaxqZpcijlQwjG+6R+RKGOMIlrAo6AjgIckAPBAU/J8Tf121juXDTVM3lLlf6+/QbsGbv10ZA";
        sout << "A+Fij6lhoUS9Y8FhTihzhIgRVlzAQnmkEyEE3Xvg9lmhk8FxguvU9+FgkD9QFD+tMtuIwtW+YZCB";
        sout << "nqqlze5dTgptRiJ4KBipk6NVgq41AIOtY7aM329y7E41Aqns8ts/mH7pZcRBJfVdb9fOYTv32fip";
        sout << "UjzjS6mwnZVuCD1REPZhREO1ozKMmwBSiD/GmZkRyH6tU/DznqUjffQ5fqtFmVOB9f7fEj8J7VaN";
        sout << "1a6eLE+iUle5axd1zZGTF9tS6v5lxv0yF4V9AsCetnkzqkyIFToF/18fWQC7KprRPsuzDv+AjqZ3";
        sout << "53i00GIyW8FXoQ/1OUZ2cwj5YUUE2ddOMtVjhTxDA7ZitwUq3j5SHQT2QH3TvTooskOK+VJToIs2";
        sout << "QN6sqHKAX50UnO9szXH7FRsk7QFhITqgfiiHqgY8eJto51g9iyeNoGxNO3nQBzY0KhdFvBFDUim+";
        sout << "KCFwzYOQzY+rqHGs9z1YBET2m7DHwrWk3fjMuOaeDN8c9cW/kV4eLASs1doGSxOZ7mU+gOrG5aN4";
        sout << "9sNnbS/vIn5dW/szhUyZPInXKiUJhSBEBO2U8VzUueLfMl0B+eRpENhg+2Q+X6cgiKk0FXOdAT8S";
        sout << "mO/ZMJfhRVIzFmVJLSC29w2gIbh1YVWEi8yN70u5HOR8PPOOJHkUszQWIxgp8+NP9P6J+u/J3qVk";
        sout << "wU6OMUzy3dN+UkUr9z/obaFERqdb6zsqSnxv1gAzYewxLMpApRmPNx/cbdhsEP0cOibgIiKyHUvJ";
        sout << "rbv2srCbiOK/Y+y8mHjhQz2slXZmKE45v6ea1TMzTW8VLFpj5pocFmnG4hx1/qDdTHEgzT7FTCFE";
        sout << "9Q03J9NHMb9wNN2+rnZoWGlJ4fOiPqUTOvOgrlXg3lx603vmsr7vTsoySXGWIQ3E85sMQw2XaSQ3";
        sout << "qxHs6hKPJaxtitzL8njZwmr9xonh8raZ1YTPkm0/YQMRoOQnS3iYXfriU03T0gW6VWe2WuhQRM7C";
        sout << "69sejg6tLZP9/XZHgXv4CeS1pYsyGitytHHtMY3mkHhaVP1sodQ0IBkxDDBSWGKWhY54aC5wqMlM";
        sout << "hEFf6rBVSqZGSknpZK18V4P3vcG9BuX+bzNF/CHabkYIjZD7QqakUP7sYEbl5KqEczd/hhf3+qZW";
        sout << "KwlP2f8HnyewuMn3zgTZi+tRQ+vyJ7L31pmJiqB1j6i2tC09YQTUQ8zz9uFLScBahphm5sSkj1N2";
        sout << "dCtPLyQWfed1PRAkw31iwTzfOJYedYOIVU7/ngB+GLDUf3YuEWXvN92UM80Q050Ylu8c7MSH8KRV";
        sout << "KV0YcOBPmR53wfJimwH/+l67/o27BI7ZAOXLauoTm7hyntXiauGvQGDx1Mhxd2BeYnmDnXfPZRW4";
        sout << "XaDhz1CcjErR7HWSbT9hv8PAGtuZyP4qKu5xb5GkUdXLIuCgeaTWjPkhWoW1UVVP81VWquYWAkuK";
        sout << "KtFyeOm8fIJUE2gNJT2FL9HJ/WNrsRyN7G8ggcWNXyW2tPh6qBYUVR44aZ1sQ0Ejf4iyP9LOqvo1";
        sout << "MY/GSx13lWXhWQ3W21GrYldCbIA4axSwjmys61Tf+4z0C1v/OuTp6c1oRUk874QGtPQZRbGGFg6a";
        sout << "B4UwwMxye6a59fUGg2J5e6T9+2+9IPfpnV6uqGwxOf41qr4x0OZ64JhqoXY55mSEu/8J8E+QbJu9";
        sout << "dth2Ix9zuNMRTET6BvwEPIWGLgaBdhP/pi2Ioyc8oHRWfoMa/SRIrwS5e0+j6WO8MLd0r5e3yG+C";
        sout << "B2egRCDy1t9Vn/mHsmrGj4cBO23BQm08JWAPREd6KQK7S4kJ06vhPYYWpP4IPeWf5PgPCBqOpO5N";
        sout << "xY6osmowisSognB7fsr217enna4NmIeTXFywKIOq8cIvqFJGv/sxkzmyR2P0iNxKm5KlOPXvVy+/";
        sout << "KCsvKNp//ijW9iFqBmVJAygLcoeuSbQzxok3YUJn+gJAwgde5tSMrpXLaXSosUS30YvmSPoT80qU";
        sout << "h29GU4Moorwb2bPjdomWSfWHEwogoJ82Kh9I8WHDQlX94oL3JVmNfybYbbwzSRT6LrDpcIUJVepB";
        sout << "cRewsAXVJctr3HyV4LbTa8zKB6nYnrlhAtP/Q61NKL1GgF7ht03T7w81eeTgAzNsEicorkxbYt4Q";
        sout << "vawVc5sVW/H76Lc4cwfa/0kT19AWbUzBuiDinyadSyxIfpYCCf+GZvboswk0wz7JmPu3IvDtVyeb";
        sout << "DO1uq69Xr6ncDbJj18fyF0aY/vdrnRzg7rEa+tW53TcclJhrG5L/LzTf89D8p1fh+bd266hNVHwa";
        sout << "QmcsHOIjD6H6lJUV+3zfTXgZzynD2JLdK0SSip2N0h8V8nQZ38LkzJKtyao0hji4zcAGPZEMUkAr";
        sout << "24RanBoizS4+qIfB/RFFOJIUVm6r7sAB0944OGNeObOHYIwxoRY7vcTgXYr+SGw14aGYoBUxrRVt";
        sout << "MVgQyW9HTFMnkQbC/OpK0B1YcIQ1Zm7IOGyQkE5RDz0RBhmLm4CVc4ad+GN4U2HaxEXqWAQaBU/O";
        sout << "Bbv4CN2LAep2M8EyVX/eVX1Rmul348HNebxtYOWRS2FahRsUrkU1KsiOqGB7O5XxolnCrQMermOk";
        sout << "yOLpm5n8OclolkFFMTIAMYBBXj4PfToOUe5RNJStmPQCSScdg1nvDtR5+m/CwqnFGEsPqIfv5Lqs";
        sout << "eH8FfS+ilhlmoZiniuMlhWK6U0CHKaDw32cZVsROyjMngmrz/npUOAfmutzvInBzcPfs9rGylCBr";
        sout << "G6JVJucnTPzAqS4RwAj86hrsumsxsWVXB8aBo6siNzIQ2CZ5TJxTCGMdwV7eDfaBnYuYmxRbM5f5";
        sout << "umG6OuGc3OIZVAhrfv/vjq74V9Dv/Tt4Ewt+Mal2ZHTCkHu12aWCfeUNpULWByv1/sEN1Bvi+ucP";
        sout << "kUDezGJLgxRU1AMj9n9s1FP/LTUofcvnwwyZfH8HxZ77nf03I/x1hgbHSzvKoU65YJ+1SsBQnr+R";
        sout << "MlVr4Z5GQwKdNnr3Q6NxUyynQeKZbMIeqRaZhfNZJqkqZNCrhrRuKOtmET328AsF4pLvSgxxgFwq";
        sout << "d8Qt2OYmeHspq6qIdv5r7b9FMq/4iJTTZS/LcmvzbuS+VxnnX+vkZ0i2DeHK90isyfdK+ZgTUk+L";
        sout << "fQf3wHm6MD/RCcDv9r/Diujx5YYoB29jPam5VxevW8ScFcVvgqVGHUEuWa3kq6kDWuiXWzlW6kgX";
        sout << "V8oM7+tHHFQni3YtBPk0EF2LhFhVeNtIZiAKdX5laf+yoie1NO/D6/+dXrvdbyRv++3wX1To3sL2";
        sout << "2RETm4/wTMWixMhuARCeAitLRlol3t3mMSgJYCjAOVfXpL56XqsOHonPJH3ZIMALezFe2OBe2iDz";
        sout << "domycG03g/EVEyeR+bBgv2/8RSp24I08XBQWgvP7PcQkUbzVngmm53+zibIiA+osJdOfRZ393Fa7";
        sout << "/g72lrryeyPvd0XZAP6dcBjdcmgcL0AoRE6YA+Ak0qzKl7ppTJRFfryP2Ik44lMyZaVdIjMdOfuz";
        sout << "tkxFyVCMUnqOxxpc/ctcmEU5GftnLBa977Sv6ss5YzKiC3m6z6ZCLND6GJF8L+ggcvj5xaiQubkB";
        sout << "mAXKTQxCyUdTCvMimb81NvudC21Ye2mPLTreTX8fgiKn3KXv8W8tCW/2nInwRjnK5YKixzwvf3+m";
        sout << "OtMQuyY7UiPDPfof2hV4wuaZZIKtsJ0kOLnRMeYKbW4eKi1Ut73+6Fo08fYOKmxWJ1W2XMbWSo4c";
        sout << "6A7VjMSps5QW8lwJDIC3EJw0b6pfIXzmK5a1z1t0hIyJ/kA7g6FtQm59elqNDbr48lT/s67S3kT2";
        sout << "e2FVe68FMbtTj0yH5zPBinjlfMvFrT7Qv2v2Me6dyOQ4imB6F9ZwG0rMRi7LuPbWpGaHwCG0B/T9";
        sout << "Vw43+sn+K4EI/7j78yeK/FMBn8HZq0YHg0Bc9aCAKLccMUjRxnpl6Ug04Qe9AArJQk5nOpGde3Rw";
        sout << "FmKzAlG9Ki0Jml7wAMHsocddgSUIyRKhbTdf+RBqdObERZU7kV2m0gydOuCgkMSdDisZBdrcbOUU";
        sout << "fmIQFovhVPlFMj1uzYxOnYqrx9OlZuf3gN1xQdOv6AfcVli8tooeV9bPPfV2LdXK6DmmkRpcMo/G";
        sout << "27/tF1iIPEB3n1ccpjpGGDRD4k6dCNTfnPOozQDlPbdP1ewAgSbaeVYQEdl1Yav077v8t58aDnjY";
        sout << "XlQZKF6JIAYt/vccNql2PbK/YmDJZ1ino5aLyjLxMryjWQzICJDFydXAjdZIT4C+U4P3zQAfc5++";
        sout << "sl9I2kgWG3M//64GuxYFq6Q++GlGWlb7Qnv2e46l1JFRJerjcwQyVCbHAMOJEs5MH+SZE8GZSeTX";
        sout << "3Rxa6jG9xKAHY4cJj2nucJFxEP9sREPI5wMrMyuKSq2cBCbQ3RV3asPS2maA5TPzu3IkkzpBfKAV";
        sout << "RE2kpoAxAtWV29UpkfYgcC5KFQBrQQs8wobaV4W/kLGmBT0QHz69Os4DmPf325yqK4NW6I11U+yu";
        sout << "qv4erSMWtWHayBLGfqpJZBLimJNWfwyTci9Tg8cGrexqxheousrddHSV5pMidwm8AW12EEiIZuI0";
        sout << "URRn9sRfTPi4WFJX6pHSrRlhWNuEsMfVIhlSH3YhpZF5L0xLGN59yXluvRxO2mbVjBi6boCod4S3";
        sout << "EXhaensBw0cCrU0rmoLcRKycSjdVygWL/aGOYt2hYFVbIkc2Zafr363PEUDCDnc5m3t44+mBDh29";
        sout << "XwmvzTFe9BOweTkvMiPo7WrxaOhdtOhrcTUlQLXcDCmk5UeAoY2mp8oovwQCyD13Daqv0ewN+WcB";
        sout << "dXn0ueRNkQOkGtpQ8cx34azPzdKPhX8RUHaAbfJVWfJy1OEXUb1MmtLt9jEdfumBQ+LvkyIWBdF9";
        sout << "SvU2Jyb7wRjUNXJNLCeVl06cU8Bb5ArOQYok7CfHZS9wPQEaWTUP61L9I8A/6QgB+diF9kdS4NXz";
        sout << "7P2aR/twV6HMrW/xc0Z3gGzhHvjJLKo+4ADD46oqDcuByc3oxvhOUJHz5BPPr2lzocEZM3FIh6EC";
        sout << "KPTv+ggDx46RFWFXxlADmzOM/GB/KbrBc8pZaLFDwR6ax5347sLZBGk+gxnklBI4etx2HC4Ousiw";
        sout << "rLYFKPYwj4uZ79UdwW122Hrr1nQjlKShc8Y5QwnLbJeUNYqSVjY4PJcWVATeJ7kZceLvqOxUQ+y1";
        sout << "i6AZZvSpixqu2/6BNYg+wx9x7uBfimTLowkoHyTVZ6HOYcLjrhO0sxs8dIbJlxO8l6TySfrhwosN";
        sout << "ihl5I0yZkFhEBTWVwyYyfg7Y0wpRk+89KfeAkBPztq7gAM67oPLCMaYSUJzK7uwf9sdpoGCOle/9";
        sout << "Y2aXLiB+ry4CeTFqIchYQxXh4G4G/V4/eSYIAsHte+XHmAIlIey1jluutOyk6zB4yzLoiF/k+q65";
        sout << "wBzoH5jpI5NdHNpJMwrjMle+56P9ZMij2+6A9upKKHoIoF1f6jxkC2PlaIpGPVMRtm3JlSXkkmj6";
        sout << "T1lGIKtlt3WukOCj7vQ5XqhCUN88Aw/qh5wbZSNi2JW9Y1a4yQ4iBFDwQ/TPX20DfPUOh3clXqSG";
        sout << "oKqovUoK1SpClgCPhnPa4xjTXN55XWfewmh+lZeYz1mktiJTiMmK26pNYKxrwqAr2nJpm3Gzym45";
        sout << "ur3c4i2TcXjraH9lMhdJYTgXGPfjSjYEWXQSImdT7FBFVRKvp47gS+kvEU+XqAfNdINFJfmXUspU";
        sout << "JbdCRoen8Cqy7Y8ZTNeQz4ZaltDeGT5lFNs0ORHwaC41lw82RqOAlOa2qPOsbhvAHa43KmWNjQ6T";
        sout << "E5HFmQH9s6MdVj7Yn1Ng6dT8hakG+jfCtL8Bc1pbxVsSznPBz0M3m3GaByiPQkNPoveN9aJEZGb1";
        sout << "7NsK8Z2TPVGrq9dGjJoZP0JSoLXIr6GuqV8dfHgtg43LP575PJMRh5BWNbOdaW9QhTLsG3HqUFma";
        sout << "cNQ2copoN40mHbVaY69HnRm6KTSLMxco8R49hTgIADR5Ea2XDKUtlvNhz8GAbywTwfyTWMnR5tCo";
        sout << "PRZHAfb33KLmJi/rseSfsEM7kpXXWLUK5WHhSpMLZ/rQstiLM8qczl+jZR32tWZ3CaIbAyEY4q3L";
        sout << "hMehNiU5nucnB3a5YcmzvNt2HgvIeHuMFQcMEwxjL5Z73eKwouBdepknIPv7J69Kg0zSzNxLvDRp";
        sout << "wczO8Ypnhkck0vcRUohMd9dL5K4itMQIhqp+x8hxlGzUYn2lf42SHM+iWvCT+L3shXUWNNfOxyyE";
        sout << "XJMD540EvD8Qe3dHr1CJrFCP40EHdhkn9MjRALBQRNewSfHZHtEMUT2gjuG5798F3uZMoeQwY2qe";
        sout << "zkGhYz6dWdAA4mARl4K9EG9DduYoOiDTckIsUYAAeBIxYpMbdUT+7u+ly4cgjwhX0TPmLECnmPIK";
        sout << "GPolYCX/5MNZ0HMHBm2+BDY02My+cVnrFBpt6XSRFFI2QhUU170ik//jKPPn7+aP47G044SUvvGY";
        sout << "NqlM01S1VU5Kx4N5DmXjca9my/rvKt/VL+UEBjwpVmMRL/8WSTznY9G+kzUbIg/1oiymovMIQuiK";
        sout << "S2C09BzNmHaRMDCqENFVLkVR+kQCsKO5BebOEzYJe14RXiGMe2xZJipQctcQqC0Lo6Fgdew4s6bu";
        sout << "l7i7mUmix4ciYJKCmg0XjD0XklZ+mgWh+1Tr3YKwqBL56KSKPGKqkkWCYnvGPGz1RFFDrcQtSAI8";
        sout << "6PBGAyWsU/dknLSlFmjA1quAGrz+qj7W68oafHRQKQLxLRky3CMikdod1JIz4V6paHYCd+6RFSg8";
        sout << "+CLfhSOCyxcNcSLOFFPfBtshM5B5NpHLE4Gs18kJygN9zaXx+FAIgFtYDYGmHPhHOF04mKY5N6Ye";
        sout << "X4Pnz8GLVHWRWsaVXnnO/yIJ7rf5Oz7e+4ATQo9SiohBdglX/P7S0ig7DkyWGf4M8h43b2eeVr1k";
        sout << "Q/XSFo/nn82hEEweKgeuTw9s7L997H5CQIX4oXHW4/TMon4olnC4aIBwBC/n/p8gaW3+WLRrHm+g";
        sout << "bVwiYlL28auMnmyfU8qCxBunr7NiOV2Mc/sJEZQlSq9K+zxzL1VhPJ+eeOBgREh7TwN0QNcQ0p8X";
        sout << "lbt4TlwYn/s8bmNyeZqkTga7RV4cqO54ycuy1rSj+8nptrq47c16TJJFmfKG0/tPFNrsg/+2Q029";
        sout << "D/03cJ1plmnoroYflDODsMqObftviKsiL3fkPs3TbOaWatemlmFhvtIwGLF1gOkR9j98+N1NAxl/";
        sout << "NaWrRvzVaSNYJuJ9L0r4+g4zvAMSgbDR90DxbIclC3+qtU3YhXJNJl1AcEt8nH+X6m6jaboKqrFQ";
        sout << "ZwEuyCT5p4ZOpvdFvGILGjO6lAvfhJyyNNnI30rYi3rOXpPQHG4OzFcuJV/+z3SYp28HTusW5qbe";
        sout << "j1cfErhXgTRoJTeC4GChKda0ZaM6ki8M6F7XyfrVVjDU7Zo8F9nvN55CCfsORuxFUVUSsqbMqqKF";
        sout << "J7dcar6xyyuygfZmBxKIy7Iwp4EbqvoJX2FtD9mqx2QSiwtTLitCIooKfzqDNly4if5w8HX1H14p";
        sout << "eZ3i+21JkZaSxv64i7kfT1LcVq7UBzmFxyrazmJ8u/dgumLnEG146tnej5YEBr7jsw0Lud8sSYx5";
        sout << "zw6evjqE+O3rHwvOKPs/COhLw/x5602eicqR4tuw3m1ttmZljCS5QNvNv/2qxWGX2jvguFVCOmxO";
        sout << "sjYL4SI0pwNdbK+9R4fpm3FYvQD/Je63BXrLLpDFDX+alh230HnTb8FlG5JtwAJhZRBjtgoIbpUW";
        sout << "1lSmHzEecvNdkUCSmC5XHepkT72XIeHGH9N2r+DzMn+7IXXTCmpm6IApP6mhLrsGfqrWXRGeAhig";
        sout << "mJV6YwCgDUzNy3A+3wxKPdf8bz14jhDcVx+6oTRDOcnhmWlZxnAkD/S6iT5zxo4PSYKAA8KnnEi9";
        sout << "AL2AKEFclCVVPFXXtGSa6pHwZVzUCu7z78kBO7cFCR3opQkIjSabRhK+OPRuhy14Mley40dqqgqA";
        sout << "RaGlgARktw0xReIEfu9T6RFdBBfS+1tkKbBy7Wu9DD5xu4kxjL00brtks4Hk7dpxLHRgGhN5Pozk";
        sout << "uL4aMqZYgF16gMBwX+OcP7zpY4xkwdJQsObWjMiZwpx4PK/c/V/tSeJLhj6QSwojJb0kf7EhLnbo";
        sout << "NbVGUDRPsww1DYa6O2XelOVITcJX33cgDHTkTj9l0bBZ/L+KlaxPsJidNA2wFWfMhCjv4+eRhtQk";
        sout << "cmVjK3qBGBfUoQgXPC+gduHdxsCOWvE7W0LBLFV0212PSyThBgYSpR/a1JLOxl1ulbpRuQXaT9hi";
        sout << "sFnTFu7EzzYD5Jouyl1Q8TLuT+EvPNgFtIK7v5R25b3XmsjO7rSkpiAuLo3IPPRYC+mB1KtVhjY0";
        sout << "Aif8w+46bbVgybERhHV1eaeD7C5do+h9ulNYcrM30a3Y3s+mRXmAqb05Z5iGTgP4CaEEnLwBnxFc";
        sout << "lLw5pIdjBrbwPb6RyknbHhYkSDCN6wrnyHgmXrSnd+XdZTU6DQSTVHAP6FXhI+s6X+CzdFmVP6Yy";
        sout << "nHRjsgr/JgfCyopX86qTQAZx34VrikbOg0/jmKu58HCnQCseGBzAbdiV+G6yGcrpFzXusAyihO3H";
        sout << "Z+LPCODS2lvvynJPELrKlElGy0mB8wNB2c3AJ/b9NMu+mrAPex0cUOc+v4gMgOaV2CWBa25gWf9B";
        sout << "8O6Kvu+Wxi/RffzmG4+Zpi38j8PeNzndEVmlF376jBvppjQw7xsbgLmCpNceVMQgn1oY2idlctkJ";
        sout << "LWs/PwAq4HAZYbqdFpGGLgDFkCHfqklR3bS99kQh2+rCh/xDk2ub1Ylm59I5nUsn+izGKHf3ymnI";
        sout << "vVuJ7p4qgMOva2Knw0AkwfLlyDzfIoV+/B1pf+8gkarcVOVzaEpbLdYjpg3/wQqLUIHwfzRWgCZ7";
        sout << "+f+gU0DnnjCX+e8mvjGq8+n/S8UXplUtyQOZeHq7cEggm2ctIKKX41CSLLsI+ck9jJ9ZiYftYGFt";
        sout << "z6zO9vTJI60UoU00Lc+80lmAvliDgDO/N19Rd6r5P+IkyOw3ZFbXN9RKtHcZ4I0sjAMULMdr8crb";
        sout << "zXk4cL0Sygykupa0SHl35XsHxyhNOCiYKpwGCIdTKHO458gA+fZLEXttn1IT8T2C1ud2Bj2/olnH";
        sout << "eXPzWt8bvf3R1AFUhv5Be/WFcWgjnwF8GasYerrnfNXhLT/YMC5U2wBau9UR/HF9CSr5GoUTPzek";
        sout << "rMXd+CVDciO7gJ8fqMQuA9YUlWwjpfFUZ9nzV1scTrhqLXjTdwOeXfNBcg/DxKvcdVdiVg2vJroT";
        sout << "QbWRhDNYWUH7Hu9p2bapjx8CgFi8CFk5PEqr7kvuIvmNvuj+Wxc3KROBmRw+Lr/v99Y5rg/ne7Vm";
        sout << "GRpPwExR85yHZ9vkgDctTv5Hwu5lEtwyET8y1H7Pb3Z4qMv707+TGtNqTUJ223bhPDb6QKq8cO9l";
        sout << "DkFfzXKtS9RXyIMj0ZEzJdTd5BHDBYuBsMnl2ABDzRfRMLQamkfyAjSluRdxZZpcPbmBDPdvm7Vz";
        sout << "Zl5uRHIY+UxrQjXg111vFStE3mtcmbHl6lfT1JfHpI9CGRzJRebHD6SNttdY7UllaGpKGZDlIMvM";
        sout << "QR7rFUOXyr3xKLvzbjeliVgkPmF+DsdFIKEY8cA/R92lhdsWlr/SJk+/IvvGLuIPQiBvlH4zIWoJ";
        sout << "qTAF+c0AX+qyQkbZoOO9tECnf2SXBp7alcnCkpdmqikG0d/Fynra0UQdOGCSqbcsdyDfccTSMdtc";
        sout << "wOrhH4y28iBSE41YmSXGAPH6wUqFeLaMMg+Lh4jrizaRf3sgd7FLi5fGaFYiXQiREL1wL+uolnBj";
        sout << "75gq+VFaCvMuqmozluQ/tR5DBnAFlahbhixs/jEAZOEEn5R5/IAmmwfYOQtEXY2UeT4Zrsjo4DWZ";
        sout << "3oPnwzUGF9fxnq1RJXeTm7eMr3q6EbOo8wAWp6KAhkNVqXpCfIAHMm9XOs9EH6cLJqPLQCzT11oT";
        sout << "2SLl9nuu02ehP6j6c5UKlV2dUlhHR1WRjpCfaHnm7apNtuWWyHemKZQlkkmNorMRGtAtvAOkH8HS";
        sout << "SbxDspE19lnkl/yOzA2pm0jPWN7RVW1gKETIegOZy/XBChAA5oHA8mkNmO4/WXMzN8U53d9sDUNI";
        sout << "8Qljh6GoXgiHgOFIFnvC3LipUgvCBhCWVunn8rSmZZ8Gg1bsFb/63ArO4Tv2B74Ia4z1YtCxZugd";
        sout << "lgXwuozlaK3UDmOjtABJfpCpNDfmdjC6B97VyqF3D/Txyyw4LZMHvJFYFyt3WQ1h4TZfjYUxV+ZD";
        sout << "n/xbyZi+MPoITqUrUPFfPwPpZAcFoUER9yT3Qh5vm33PLp2eHxrO0n+GUeyO6LiFbuZVhm4pqOoG";
        sout << "5VG8qaCzWXiQj7WWhv4Ng0yz5U/cn5WT+DfXlhVlGRkRgKFHJ/c6rPvdFRx5/V61UAeEoPT9CoTk";
        sout << "YWtGm/j5vJQp3+fsxzYSEg/5ehVu/zAt6giUbL6PX5NEuHIHLpzD2lY4WXTqJgqxGtWYW/6+760p";
        sout << "N1VIWll3GG6aaehOZbB5cNPs6j8z+uuTuR3H2pMckZyd50R7dcj9cEkuFJIVjcPVjEXUMBM0c1Fq";
        sout << "mAMEW+SAl6oGn54ZdtteN/ryvTdACcFgyoou9pi+iz5QO0/fibBAS6xBIjwiTSvwh/dy+xE58d3w";
        sout << "45a0ZzgGAFpYS/T+ekpntEcUAZDAbXNrz56M4GrgYS2XQkdCcq0t6cMnFI63QndWlWe72TjF/G8j";
        sout << "EPL1wCvmIS1lUCOvkY7XcqVFvVYurcsrAb5X7Tq4W0iJ5Q+3KbsAzFaT8gcaNMNDFWgzPVn6j1TD";
        sout << "ZPvIoWiZE8DhtT7F6Tv2vNwEoy1SNPJ+66qoWAR8ylhxaVLy0q9FM8Zux2JxsDNsOs6Y5gRuuW2h";
        sout << "TdVz6NgZ7ODiCteUtc3B1Tv3O6jGjEziG+bKl05Nghj6LgPoCav90Xms0Py9O4tuWBjyivbAGW5d";
        sout << "cg6kRLfg3ziNOy4R8grrsJ3ERGBoYZVZ9U5Ws7m1e6hfrzN9gryVX3FNE4dJ2SzI4zguOWprY3p1";
        sout << "aVBsLacZQOqke816D+tW7zjUTaUIRAA4LfQA80KDUko5xZJaI61Tnu1cwd2m/pIcpkENomF2hBqo";
        sout << "IEYS2mz71v90AgUXNz/2+bcpCDyCpFcJxOtHO9pWix3V7/xMGKuHUTtEVr0ZsoMBTYyv6R+4GsNu";
        sout << "thkwGqFupQ995IjSHjX+rLQkZuwMBNgktRCVrNYmGcBUysIS0w+EeCDwmFKgSSZg0PDgi/TyrZQB";
        sout << "06qMl+LTnVqcSUCA1BKRD0uRJxUuoNo9QrN9hmk3K5WdtLV3q2i16bb9yJhuDqcit9CRatAg1doq";
        sout << "KviTjuzdQqWuzIIwIFV2FAsIp/oT8UxKLV59/FnnBvnj1oLe1Heh28O76UfySBm8JycxvNXTUumO";
        sout << "6zBz8M1CyvkqS7/AHMN4WVLZt2C50nsEs50RMJT2BnSdg+bAXCz9ZYTXg758mc7x+wy0m91P1/C2";
        sout << "5zMUxw9409SQXc5pyH+1jYnvUJoP1e5EfkFQPyGWVjwBomiSPgt9u363Ga0Dn8ogx3ndi0AJhcao";
        sout << "6dDCjsjsguyh4coO55Pu/yRNPYujpdkZUGDJun/n/H+hl3D1kLgL6R0dewL6gnu1IUf/ZQ+X5MvZ";
        sout << "wKpqwyCDNL77jts2hLgVTFbGqEKj3WpYpOOPwi1hqbnfj4Gwbcjs/faQaraf/66D0jxV9jPeVXQI";
        sout << "eOE+GoXVKNKLMgd7tBBpF3TOe0CjeMMBVTdVrg+z7zz8SudH5eZ7KgtQxAclsj3niI9oXH8qg3zY";
        sout << "WXC9q48LExc1ac9AuN2vqQ5F/Nq9DchYecxhTHViTqkKtz9xzkDZAv3FwFajlhnJ4nrgja7+5N3u";
        sout << "pt9Y+i1GroYPcg5BzVOy3Sp1aoDFyGz6P/LHB1tzfVGrlcl9l1KCVhGiR1fnBEYFE6NaswrP8DkE";
        sout << "tLGZWDwJQKToAlQmzSM0QOQVGpkyN3odWTziHrP+R9IoRqTi0LGLhpV/1H/1yf6USuWYdPGBntTU";
        sout << "fLwwHXUo20qbIqRwaEmFjmm0kNWfftQqa5yD8cvm/UIk7y5CnKM1eyymg1I/TUHSrYarUDDd0n7s";
        sout << "LCoUY7AQ1isirGxS4QzmvyXP+99kXzTPH/Tc/Jc320NECPVZEPReLOCRXOFgcHt1ZL2K6Jw/Gx5r";
        sout << "jNgrqififkUljGbLB3TkfK5vTlSOaD/lRUgyYcC2r9BPLuyTGHO9l7flsedrxXhDWKW1Woy5j4/Z";
        sout << "VyibrDvu5VO5kEoa1NeU9lRTKfi1jcpW6WPdUdY9fbN19XIPPPqhEO2tHzpVikq+aqI8GTt7lwTl";
        sout << "7O92ZyL/gi/dEa6YjI5DPs2o2VeBEa06KS8HWcGsxazpl2YAMIxII89ul2xi0QAwu6T/Pi8DSHn9";
        sout << "o5iH37q1TTFfobas7zlp9Ol3BV/j965LYe46SfdJo6J6FQMWHUe7IXYBnupBXp+i43MUoUu4yrjN";
        sout << "EWJmdoeyNae1SLYjwjzQ4O4raOOQ5dWDjRQccmEpTwlLk6GR6xRoznxY9azZCqSX/esHUfgQ/Vt5";
        sout << "2XvAFWmvnoZU8QJf4z0NO7ZzkGwIjuQzWMl3vvB5gAmz6oJpaI7EocbvZmmKKFgtbDbKyysGHm2f";
        sout << "CE/5kY/pJ96UKV0pnyX6PgutAYJYCc8WjE5wiK9Ym6784+e0hWL/ukj/P0CW/hgIoSwxjef9Booe";
        sout << "U2jgjqQkTcFzAWeXlTH0RrIpgnE0VJJZOi+hSVinG8BxyJ88unIAGuWO8HWfSPp0k9d87izaXwai";
        sout << "Mi9grjFx6GiBC3OTWTVpvXA9uOZXTL9vjR25P76QL2ydN3S6eADCKVx3D+rQIOGyxM2Gg0tfXo4U";
        sout << "PFHk6kNgvVE3QqCtB541T1EYNwUlB0g31cRlnpcpGGvtv4Ppz/oXDC7NwHDKfJDBQVUlm/vgacNP";
        sout << "IRjr4ffFINetbhyo2tqHpB6IaCUPB5kdEKHhrqu0KI5WGr1EM/7B8kkHI8VxvfexSwzRNIqpmJDK";
        sout << "ddMxUtO4AA==";
        return sout.str();
    }
    inline std::string get_blackhole_qa_pc_compressed()
    {
        std::ostringstream sout;
        sout << "SF8y300lTrIBNs8mIAAxWHixona3MlqGOe6wN1n4tCfjmGSnICCv77sR76Gg4v7NbpimSDKvJZqg";
        sout << "Z4XUeNudYt0O7VfvKOD0SVYjzuV0tep2aWr8EO472Vp5nMdCdbJPpar+Mzxg0o7zMndD2ys0cdf4";
        sout << "WVVgQcZdifdxC6pWXVWIibQrEf63KUl1YjzkaOaukShL716r5QNGoAbUNxRlxR31uizc+nPu/ncJ";
        sout << "+KW68qBzpprMTSL2gwOja3W5KVYb899dEEKlmweTR+03QotsFHCt4AMtdRRXkwGswWbS0GazjG84";
        sout << "ekSAVHFxGEwKhW3Y0vYD4ADpKMEcucmzoAIbjDP3a+v98IGqCq2eYSqoKHPrHxK16l3PfYTk1zG2";
        sout << "Vr9eI4JqQWMrp7t2Wrbdn/sfe0fAoYNson6bQ0jSTLXiXnG6KwUdb1t4x/emM9TQOwFmu6Tbirh7";
        sout << "+ck6MnVKQq32IkNZdEofv4p7MJsjdiFQkf/0ipAnxGCkU6pv4tTcNBDhwhLv4EzLuM1dVIrWvgNm";
        sout << "3xe4PS4UHb6JMmUGE9Er11UKTsyAULP8dK2HU8Muc/poYzXX/b6VB/X+JHVjlkaFNgBJOlUPieMh";
        sout << "Ue55p3UNCc7sF3gG7DWTl++whcNrHnVU38T87Ostyu3UTJd9FCGujA/PWMcDlmDRQ2aavUiHzJEj";
        sout << "9GQeTr53avlyWy+R64alA5lcouP/JhsSf0+yCl5P5Zl0+ocbppCasUgMHKJ0hoSihgWkG2CVooxd";
        sout << "LfpZN4psPsgFT7j1F/w/2zr4Tw0W2r84hTBrWyCXPcVWQbhir8FXby0nu34anDL1EmedETAHXMSG";
        sout << "pK//N+FDmh2mRg1VFUSzjK3UP/xZvjswzkx+1FEjEMFBeQ2jzz4iULcqPN/hhpv35uLGwx9CGgO8";
        sout << "ZzT0sbr43vZLGaYtPv0p1xt3hS6YS9uh4AdFJc+/JWPXlS1mOPM0IvB+f0a3++19esi6CraoZkJi";
        sout << "gJQqItVI37Ygw3gz/IuFlJZKUBHgQP3Tnjyt5Wm6cy7zfW8gucwy0S/p1iP9wGfvRKhEkk4Li1XP";
        sout << "zznFfTk0wsRoUuvck74SbP3e7MpOHgcRf5iDfh81RU8IZy/LZCg5xNyaV+JgzigBUwTQP8LyrIOX";
        sout << "hfus/ut3Lb7N3x5qipf2qbbkj8l9HZfA/OUOgPqjLEZmB8FFJ5sBAKTIfO0fOCzh6c6wUzRqVBhw";
        sout << "sYkJorzHiGOY5gTnxfrFWMIT4sQEv1GTraUdIeyosUdmzfIwgJxxm15ZN+M69qlZsQDFg7ovsu9a";
        sout << "2gH4utcRMnbFk/LqjvZNXFQg/DSAiwLtvVWgePB8wav0gerya3EwwBftncPWxFmxvYpE7FyyDIDx";
        sout << "icbqBl4kIBa3YTJqFW23w6WXOgEIZeyVvc4zOlXgbLPRQcA+RJU7q4yeb+vB3U+izvNsb/oa9/DL";
        sout << "5etwREolnFOhHygiqne8Rv4Mh0yH62KrBMbEm7fG3CZpo2xcv3XXJ8Si/lk8V2iu9ydzDHeqzU0a";
        sout << "pBnqES7rNXAcZPgcwoVko1As2MeYtCtJmxI4DMmux7jNvqwNPWjEx9JLA5Y2WAB5QtzJREMyc29w";
        sout << "UP+r3kmyUurXZA4uBYGytrgZsqly2LIdppt1S8WkNG7NgGbrD/9sxMLRtsKDV6yvbESvuCWeM2tn";
        sout << "6u0gSWk0u/PBvfAAwCD8P9TtuKWe6ASlPQagwt6gFKVCMghOIVADR4BcTmvKwhkTpkyl4NxGAiIn";
        sout << "W1+uxVqKW9YoFfwkL4Xg3rcV3G4D0+NYvDjPsclqmAdmZgRhL9A6NC43bEbWnmT0gxK4Z7/CXGSl";
        sout << "xmPSw56k9de6MQTZtDgHHSlqKaltAn2ygFl4UlugpuqFhAKnnZdESHpiZm2YEETLs8ziA551TR7z";
        sout << "xJApwO9sfbvfLAbI3kW6ihM3YOwWOjiYbTymO8XbnExKCxal8S3A6Y9FupZ1s480kOEJqm2+Nxuc";
        sout << "0C5kbLOKBVJyPKQbGRPo7Qi+paonvSNiePWLvwGUIBAojo008WdSbOme/aXPFww0p/VVsmsVhIwo";
        sout << "py02OEkOAt4A0dAXuJDNdsHVKLn0SSFRi0GfNzifAUFdpDa+bgz95pQt+uaZEsiGTPYgDN8pmQ9N";
        sout << "ZMFjmrdgkNly3P4RwKE7CvGN1lYAyhy6wqEEGadUKYRjwJe6xiHkYfRRqutuXT3tJoX9FTsGFvTZ";
        sout << "jc6vFxrfrc65H6lI3mCgayejll7DCCcnUYXL9FHYxVyOuoA/GK4Nx4oMcKwIfcMZNVE81nIH9Z98";
        sout << "vV9GXFeyJVo4M+bgFZUeOhBhOp4Gqbo2rCFv1ECyV33wlwZMFEJcMP5U3fmiaKXKxUgPEwANDg4y";
        sout << "snInRlnQLq0Hte4FloegQr+3/i2f5CiGwJY5XXTfKQnmAF94y0r5H2pZBB+GeLTOQ0yo8z/IzPfi";
        sout << "wKL2Ul8aRquBOusLTZRTJCuWD9VioF0SDGy/3QGjf/iiPll9YolK4oUtDsu6EI7WPawtwzi5GhtI";
        sout << "7/raf1kgEKYxWSRwZm8dillmQvnqYIslGEAfEN18Bh7q87xva3bWyPF7EVThkbpMyRoIxtsFoi6T";
        sout << "ULnEojSxK7SBodXUiAyQS08TR8Y52JbdsC5i3FzhvILIA9avLN+e6R+df3cCrxXLiwdLh4X0e+cQ";
        sout << "tWZiZmzDBCUWF76ai7v3t1SWvgizkOhuI0crmav0wrKLey3hJK3WWVDrAc4Z5f3o1MFM9OYXKcDJ";
        sout << "aKQ4fGlkJ5pl7ssmo7T7UUqhTssgIGqy7xVSJoRMWQ7V/zr2YDw79wwc8KwXRDT1iQlOJ7MmxrHt";
        sout << "3Mb6uuDUhlX7M0H2KvhOXT5+OGpAYtYevuYMvyhYb9VNpFPozh9x+oivyR5dPnAxdXxa7xMOM6V/";
        sout << "TlD545KR89e5DZtWcwCpDJgRUEHaDB0P1tCPzH7l77q9cO32DNCaTeHkaPwLFjS3VWaNnB97SRBP";
        sout << "10Hti8TYREZUstxsiWbEPppBJD3sKkkr2TpoU7apLL0yd3Elm7VhI10QNBXy3VoiNWcd7Vn2YFbV";
        sout << "49t2d14jZJzdwcaymOQ7KbMmrT/xa0azz4mhvD/wSXPfYwkocetIUYbCuMtnaS7lBlhWG0YLyDSn";
        sout << "ACn1RL/lMrrYdwOVJ49y5RXnwomCcTGZTc4/CzTAF1FYphoFkDMtSMl4pUQN5VJqRX1sfeSlgifJ";
        sout << "AD7yYa+f9/Qc+wqYn8Tm8w/VIKJ/L4W1vOJsvUSDz91owE/N1pTk7swl9QYWHP6N5/OkvC5EukxW";
        sout << "E5p3I8fu+m9PWy+vlhbCGDSTr3tSEUP2S8dBJXIu61ZjypOCvo8OTE6dRc1CV9dn9JhmXmTGGtFU";
        sout << "qq0Kn66wCEBIimz7hAVnfbTEmXPnYn4qasHgXd4ZPWD47hTw6N0Ub4VCtoXpsDII5D3N4USI1iGy";
        sout << "eGVoJbJg8MicTGUQvFdHLBP/5K+Y+pCsnXWSQ2AiqHfPvGCUVk7o3MRnTsL6xNnnXcmioalesINr";
        sout << "xw267OWuX2LYDxCSBzh6dkHSpQ6oqJ0g0WD79BhMcJ2s0AdX8I6hMjVQZfQGs82aTUekj4oSzxE8";
        sout << "Oz63bGkifTsIF7l8gwEY7qb/4C8sPlw3ZN7e0QKgJyFm/LDJ8sjW4Dp2fnh7dD46zdZ0sZJnIy4M";
        sout << "bku9PI49j9R9AFrwLMYXP/bC9sDWDt9QXbFAyWRds/6tAaXaBbrlcLosKUguklj1VbxJyvWMk7Y2";
        sout << "/zeg3xHuNKmmLJa6FAFtLdLEqoMfpe4AZfslF3jRPIBLgk41I0t7nfxbrtjD3ZHGtoe8Vh6guGgP";
        sout << "LBy4jzORb+Rp/l1167kIfiyIzwStLlz430bPyj14/S+cn4fcj+m5HntqemWaFZwwRjDrCo+Ok21s";
        sout << "t6fylKyXAAp5X1QQ3AOa9FQuZtCQPEKyzG4aD1XHDUPT+EtmNYY/znpKw38O6z9s7Gmw7rTYkWa3";
        sout << "gyTKIs7KpOdTJub4CMJ1HgD/6MZ0c65MqzSDAQYSEu2w52gb12aFxwkyMxlBgyPpwmj6hXROsQp4";
        sout << "SQkf1Z7qHe6Ti2jk2AoDxNGj4F//qWtRrjrwgG22ClzCmiFSMfbfMh0tSxvyqwBOcnZgYMOHgQI/";
        sout << "BLzDWd80hHSOMbFp4uCwneGxqBg/NkWc+5ALgcSyVTwmWQASeMxtIXZqheST/uvCBFZfRiGGmRlm";
        sout << "aDcEoDvae2H77m3bNJBqSDkhEoabGkgBfXQNIoZaTAAYI7LLei8DEbAitoKFWEOWYKd9KRFmKvA+";
        sout << "gatv4tUgHOag3S0zLpQ9kTWeKYI04FxldboVrYfHMWveAJDRPfpKSRd8+h7qRhWvRMX3o1XqDmNU";
        sout << "T01kP0CUXi1wvuEWqzETGn1A2iXSdOaDkqT3kX78+UuXDZp+vF7bE0hKolKWW6SKk1MdzNMpxIRs";
        sout << "nNA1YIEaOIYtgxMu68fA8oa1cOxT88cBunrb589y0g2r+Xf474nU48wigUEtRpPVy8h8g/3d382w";
        sout << "NOGurAyoqjTfO6tPaCnbgYwCc+YP4Xr6z+jRGuGf4Zkka10DRAyMgcj+4d1mck8upRghqKxUS5ws";
        sout << "GotDN3cEp/6SWVMP/vlk2RLXtmgfrdPwRYbwKa6fTJaxwAZFzlsH4YXQ4lfaj1/NOlDsvgEevcfH";
        sout << "tEFRPTcH36z+Wpgz0XMifGwa+n9Bih6XfihIVQgAlCeL0zZQSzLd05pq6jEB2sAgnqdvm3o8fEcA";
        sout << "WoAG0QkoGn5RWVXUqidqTdf4mz9IEDpPlVn6lpJXlzonGWRLq8xEWLnrezAWtBoYw8T2WyjlIVEK";
        sout << "hbnIsgQUPGpKg90zYOqeWjNsu7jImywsJheSnTW+Ohdjeqk8Xy7rMkV1UUV3nGPW3s50aQNKQ5Ep";
        sout << "wq/Gw6dq8K/AsfKrs3XuqjR6/PcRH/PQ04+R/LokFH5WYS9yG+N5PXtX/87O4sZC7FEgmIla5E2s";
        sout << "G22R+60/d7Ilca0ICIFTyBAgMD9t0fMpSGDQJgAo4Fl6Qf14lqBLbKxM+7Nfjb+HLguAKHgGCd65";
        sout << "vPIXAraMFGzrAKc3Yj+mqPmFO8Lfo6H2oL+ysw/hAHx1PXlXK/KWzQmKDgWBefF8Jmw/zkS8PN+e";
        sout << "tZyHTR2xOip2KgZQ4SSfPjS2KVYKPiX7FRlP7uzhOuv5fAOUlThLooGgLDvyFTtsZFDfRBESJYZ3";
        sout << "2XdiPP+J/sL9irN4d0K3f9DLYbKM+s36UElUOaMG8+38sHLjIEf0VPtS0oHnLEmqqee9qYqK0WKN";
        sout << "fw2sHP8F7zNLUHjX6M+ENm0INq2z7HCZHj3z+Rj8fS+VzDFzb3gqvTSfkgFNxvEDzm+bJnpaueU7";
        sout << "k094DEZFWJioiW0ZPIf/xec0yfpXFFYZo6UTqMWV8kcdEcCcOJw00HZUEyUk3EpI2cFHCzLa7jCR";
        sout << "Fio43tQ4tPNlo7o7UEpDgzVcb/Qz0aLhoVuRd2jbNjKUOVwBMzQlSYcbs0Q9Ki6+bYVEDMMCMXa7";
        sout << "MdPaom0bfKAq01FlZ2nYfpeCUOTfsRQNN82kzMOIzK9ZFrqrbNkwy5jXOWWrfTQC6IAzQKzHV3LW";
        sout << "+IL81YXjDs1HgqvTaUz2svy+sCW4Xe1G4/GSHnhdFWYtTaJFLXwN3iGE55TbAAqjo4YNetu4W+dN";
        sout << "8by4bPxIVK2NACvmvxVU5Fxe2CvAvk9I/mO5haOrH027PX2mCkSI2x+NAB5PV/3XfmrCYcBcvHwu";
        sout << "QqxfpbrHH4hUocJvCed9Ed18k+egLcVE/gK3vstngVat1RB9R3CY8eBWRIWBhKMvfb92xC74vkcj";
        sout << "9IgxgiM9yvh0ASbQOH7cGPMoq6gz9MHto6/Of5X1WWJx2q98suh/m0V0bO9NJvHZB/EMArs9u8he";
        sout << "wjRA5Rb+zPpt/Jzpxp1Gha+PJUpGvC7NGr0YtgAsj01BLP4zddWg/ETVpjIzYjn21YzrzIcWPlGj";
        sout << "IeCbI2BCyWaG8S7RbL8N/TWBO3w2zWOrlgpUKmVVAc6jjAX8tMjx2yczPbCJnsRys+qp5bCCjQpc";
        sout << "gsuZ4JdjRmsLY5ZApnUcv81+em7Xg6WV+9GlZ8kn/ueF5PZJkSxSBeEmea99/Jt80qn1EDd3ARO6";
        sout << "7QnwxrH27cZ64OEqnCCZMWlvFSir+xFh88btShwzwYajeKol50vjnx76wlKhE/PxBRvh5/GU27gm";
        sout << "UiweL97nHIzB7uX/OhJm8ublp6mEsbQLtANtvkHxcaQtilyVCyJMEsHftQXvxUC+RNL5ZFS+xuX/";
        sout << "vzc2W0JI2HVMBoXQLH0S1ypjJTRSRvRINxm+vYCuJkzlRnNuDp17xFADV8MkaHTeMmr9k5cCneOV";
        sout << "/BqGDlzCuteIwq8i9sqIOxWqtVARGa/LEM/AbqxFCKfxtkGnIhXh9nmSpzEI7O4/yE3bbGD9GUS1";
        sout << "GoMW0jGsPAUD9fv2KHphFoJNJ78D7hLGQt7PuLtUC+DC4vi6INetfhLBNZs3EtallhPwImzmnFFt";
        sout << "pe3vyCuq+VgxuFHexApw2iBuE31YV9mH8orpULFyAyn3tklEySa8hufNbR8wlQ7tNwFmjXiGuYhN";
        sout << "mVtIC+K+ak0fW+y+DcEb8PBsvATb3y2XZNje7BzqmuAzAb+sfnaUYZkLnIptsRluQcR66QtCdrP6";
        sout << "zFE31twrTMzNxn/3pklniwCLEclfap2Ivo30k+yWGCOf66L+eZvPLRBqbC3wsFaZtlsRY8A9jcsw";
        sout << "w+HrdKo9xq5UeYV2bomfZGxA2n1WFFTGWrXLhRs5lxTd5Z+7rJL8sKffI10ktL4ZDOQDahMpy6tZ";
        sout << "qF6lcxqioAi4ypbgFODjk7XNprW2nwpG2tgtFqqYUMcc8aKwmySpg742kVwau46bvaoJNwMrrKdj";
        sout << "f94x+9Wp4TvTIJBoACzqj9bU41EJX5JvK0a2Ak58d5/7thq1gx4n/EgufTAbY7zDmU5Q255gisfI";
        sout << "idt/Y3MLVykpGPfKBMxviWUxwScymWZjWmHeowbKFEtl4egHUld2jjS5kL9hb7b4PpludHEOhZAb";
        sout << "iN/Rthcz2MdGFujylGOH896kWi9aAGnCTfBqPlHcQhqv8l7tXsohgD4mZHsEivC82aFG8Di42bDf";
        sout << "omy0XJUUeVNqI5/tqsu3LPv/VKZ3hYKe7SPReMeqsjKoEFByBKw6FxtMDKW1eLKGl/uu54/JRcqh";
        sout << "nZgWenfgSlAeNnlnVZD2qkO9v5+D08VC9HQJ4Tsd7uhgst0PhipXd5LOIEvbxCKG76frdSuC4LIv";
        sout << "wZqPT62uIn5sVG8cII883Eej9E0sH9vrF3zdbu6rS04lbSbjAfzFo8G75FWllDZeHRf5Bs2+87Dr";
        sout << "kw1vc3EzsHIjgDV7Sm4MImYKnsvJpinEOEqvGhJtEDtXQ2MoUI3K9mw8cvcw/6viGNcICbjvk9uQ";
        sout << "9p8MfxRwLOt0DXgIMnQzyvPpWeSo8F2ducy8d6TC0RzDIS9i9TZGY27kg4jbMrHRyuUCwGAy9IHT";
        sout << "CSyAIMKFmbcSvQstpLMsQMUz4uXeo8jzl+WTIYnm7MNoa6n+VeqIQJJ6x6+I+TC17T2xYNPHewn/";
        sout << "k8MLAi2DMAA6Wrc6vnZQGe74u7kEi9q5+F/8ytziBJnGRwweaoHB8Jl1roWhtbcGxnPld/co9tPg";
        sout << "ZoBPqKVqcdOsINt5oGfcYkQiXmWP//9NFBR7+Dl7jeVAXaRaP4QGf5LtQOTRiG6wQ/xhTHbQQKs0";
        sout << "UGyX8ieBbImswl0ypnIDxy38rp2qSyJo5l7kMN3YB5u7vT7QoHoT28IGoCgwwr2LL885GmYfZWpa";
        sout << "M9Jb/UPe1/z73EuD2IzH8hFULm6krbTWjzByZ1mt/+YPU0bSv6CljdJnYzAeLG5DLpH9YyRAHNpK";
        sout << "+GELc3kkAtDHNge9hDzdxTth5QovrMyw4VPptAO2WvvoqVTzWVSqGn3vOG9kfaLp5WT8sc84fLRF";
        sout << "eqN+2ZgR5xWfCeOMD1JzfC6c3mk+J9kwQ8qvaDRFzkyoSbyrQRrKBra2G/sbxVTthQUIeuzDgN/w";
        sout << "pYRZdBOR7BY8O8eKpbvy+eymmbd9S9TzUUvL1WWyv7WJOmjgtTJkbwx+dE/SvLCAVlXW7F/dKoQa";
        sout << "6WRQ+Sthv5noXdSzeZlPXCURiNCTgGuHXZSv15uN2VxVcvyrxwejB+XRezSOytqAWaMSP5K9sgM/";
        sout << "DYyORRbhYgnXHcwbBBlMTAZnxOFiluV2si6UtsQswRvMKUyDYhxir1BnEwdqdAWdxc+kvb0hXQ6e";
        sout << "G2eR2hNNFRjVnlE+IvjHaYyP5n26kKBva6IQnHeSBfMiato1IF2sOxi9AMc8qj/dg9RhdhCaASBB";
        sout << "RIMCY+O6BsPMg2qVOdYcrbq6j1Evd4iFHozzkg/BinhlED1iEZehoq7mMksi4QArLGQl2xWTlqqT";
        sout << "F+X5rvbs566No7CMWQasUoDj2Yj8K5LfrMa77HT6c976vMLuCx2QSBFWHMBWDGPID2zvJ7tyH/t5";
        sout << "nrCFsT5AOSwrDSlLzXqQhfzm1UVA2EE5nHFrmiTSJ4lZDKgfLnAhWgAynW8ParUNtBRSQLwNOdPF";
        sout << "C8twooWTNlAe7w/pzU8992hOmgfj4egyUDpDeWgwefQAuYHLYRxlmQrm4dbhR4p0IOG6C9m595LI";
        sout << "rnkdvnEDXXlP5uUApw+kebV9LQLxeOWZNAnvUAcoCB83Y5dL4RHk9x3y93tQw0bdCRU6HiUYyubG";
        sout << "pLiOZSFbDEulvehr3BM+mbWyUd4F58tx0iLJXA3/eW8UmmEUD1PrGA4VUJefisr2A2vzAK1vvwBd";
        sout << "QxJH/xcDqNIKcAelMyUb7j8lLhtrz8ST/8hhhLeoopAyBB9B9MRISGVLhmG+TBGuOgbYGaDMnqqq";
        sout << "XUClEQleGSLa174jsrRvDGI12doM5SfUAgBdMDvt8R/9bT4Qlz6yl9XL8/qcsuty20lAJs15Nico";
        sout << "PxUkT/iwU1+F+40z8izf4l2H+UTsdBLMbEwX0udpwbiWfYDFKWewozUGlrv07oJapxEO/jphtgxY";
        sout << "lKObyLNaCLT0+NtcHvCUUzm0Y438GCVagYoF+uIgpu+KBptW9vC6rGxM/YjAsyD5zbe5g8eiIaGJ";
        sout << "Vj1IoSLxKy9y6FQkZQrAjRigBNSsM0RJWuEe1zXMW6RV7ijFcEdxQABaAQT7y0aY8b4uORpuU4mF";
        sout << "E8Zt8XzjYRJV5yPDnXDQCS/eaQZZlm2Pkp0v9L6HIaD6hEpnwWiODw33SwFyVJYosR5+L2H5FQXy";
        sout << "xiQ6HtUXcr0IXAVsLF3fcoML40E8h7IKdZut5JLFtEQRRbnnagOwV1uKVQyo1oNsnCQGNC1XF/Hs";
        sout << "lBaR2rYpk4I47tGueh4oVZlBgt3Vx4DmXjRHrEcjl3sxFEdFZP1PzM+odmo7O+hHiFMV2TgnPAKi";
        sout << "lZeEk2UZEi22mhGMlMnv96imAeZ93tSIC7UJ9HNsEvbSA2TaKwqjsRmwC15h/e98moTHuu2rG7FW";
        sout << "eM5jeL0/zAY9gcw0vK/lLiYkiIdKwkLQbqNWX6Rb25op8j4g2MLov4AZ5DwhHMq3QaafmKTOschU";
        sout << "ZQ7bhv1zJW9YTtQ7rUkmbrAM6aTepOTnXAPYC3RtJ2/Oh+QZrwhUHiePVGTXGPEHNpoQ5Ze5NHeK";
        sout << "VQf5gGwFl8jWsrZcz2vBXOUMNX1mW320wXV5Sis6avgqQ4/lAkqIi1z8X6QCBPw2qWKsV3cKjRfn";
        sout << "nIVyAsmXNPB09ZdS8MYcj4npJyI9QWIR1kIjmw5AGodMNMSD/xYt9NNx/Z5Z13/+oC2htld8edgN";
        sout << "TeNSlYRDIpD9UMcEOhnOWVT0qwpfh5f9l89ABDtriXcRrCThLsoUlH9z958azZKE/I4Q0jhvY5XK";
        sout << "GmzZ3FwGwcmjZFIav7KmGhV3Dl+9ZpmydZJ0t/rWvFgYMQey01FitikDf2g7D9oGrK96o0kI1ydu";
        sout << "wNtsaPW+BpkowXnhXmoUDRVEtHclyyMlwsP83RM/eB1LISlmL+mh7LHUZklZvmlixcobwv6qPCFx";
        sout << "+/b6YfpTit2ocTWsWnzKwy+7LdkFXCXIervGb/Uhu9P6KnTnSNPAYOSHfh3+TpytgT/0r/Kk8zxx";
        sout << "aYYaYCkQ4VlVZir1WTugo69dekiGrim84C8ibroNSyUgJp8ZHQuKTJX/LgHlYp7UDXB1n6RbRn8p";
        sout << "2DztlWzfmy1CjYTSm4auJ+x1Ik/j4Tr/4tmeJQitlIV42f+5b4Ch5c3vjVzoRyD2SxggQ4lJWzBO";
        sout << "m66qqFtSDLOAQ8IVkdwbwgzvdl+/kyHVMAjNIj+ZT3n3+vaTRhTBZz39QD1ovNQ+tUXzrrb7Fbw7";
        sout << "qwmce2c2a/LHOAEuyrZfAiHNDSljHNtmrEK7BDO1LPtp64PmYkG3AC5biGYYMNkanIEIGQYCiJ78";
        sout << "O4t+8AOr52C1GgoK0A3tv/eZFfvFTLdxuIGENOsUwXhdxtHVPeSV6NsyH1oJ5fMr6qNgu6y8PXNc";
        sout << "KOpD8C61dCvJOI8PystCoCR72ee/RF9Nz4f7KWYCV96ByUxZA7i+nX8XueRL6LRdcloLCpIE36AF";
        sout << "YXVBs99r7dXOA5l1r03m26CD6EUcDmn6jfA1or+ci3BSGNqTPOuu4Ds5kIFEYJF3BiQVGWHkebWv";
        sout << "sMtCT8eNS4Bcm9x9u/AuHaaaO7wDwl1Gs15zbMWcMwo7gCB1xLd/jBVJg8ACdD1cYcscCrONkgVY";
        sout << "UKhuVAXrXJEY34MND5zdzY9tew+yKrNNv2RJHA/zhV8z1pkW4wcZWFW/iKPFJkAwB+wB64VOcugd";
        sout << "j1RXF8Ighfsbd6b4RRFrNo0pg+GnqgONn0oKy6n7E3JUgjcAQyDHkv/hDoOmoKzFV+rbnYOZi8wE";
        sout << "iuwijzchWo4Msz73xMuBr2F+i+wzdYVALF1hp8/DG586VDUMAEHBVBXmQ7nH66WVt9wtFC5Xyx4y";
        sout << "QmXgDA6NqVapUNT+gWqAo3IBLGcfxiT1UpuHPbBZwB5q2MEuTm+Nxnw+rgM4Ay6kb4sS13pD4h/e";
        sout << "FoY5npzSkd3wdWKL5MRV1EfWPuROFNegngUYUnEhlzvx0Pkivu0+AsrincQaX3Q1uDnKjPaX1BkF";
        sout << "flfoccpDUFK7yBQV04q13mFNxj7daDT53+6NPGEDnlhYIdeZcD81YRC8Ma/tMP2QIqtzSZIp66Ka";
        sout << "1nveSaSFApF3YaVhKUjKPXeJT4bPpwO11s2W9MPvICjCGFldxzQb/O0xcTtZOJW/8U6lWz3UiZXD";
        sout << "LBHQ3PVEtwqxNip8F0P7JB1bWadxYYnLDDOsAVvXo6LFX5e2xlCrG1POU1LixKupdVeAF7UuAY+j";
        sout << "VTaMc44TGHnOKqPfqaG8vd4aySFanB4xWofI1mCE2G9bqjps/Yp+7Fazv/0LcS2l/hkNanRMd0/o";
        sout << "4CZSjRyLy0DNJiux0M2SJ7FTZAuGE6OaZXGsNBzWgcPqnZ/ivS0jKlMt1153dpw+5nNhi8/xmkVF";
        sout << "gxQIKbx3Qp9uhT2ZFkVbvmtwVTK557LYtss95TBkEBvaaEeWt9VzO83dEHNawKPYCMHgzVGLCN3e";
        sout << "7ULQxY0K6Sp5qk5O1LFB9qlnWXgURPc+Fy9H1gTsSVmwjKM+1Z7oq6vZZc66qO01J5trKfp1uAQ6";
        sout << "zJG6GKmQnv3eqiN/ZUXfGg/ghfz+zaxU/b6gXKQVmC5vXMfEH2mFlyOCkD/j4sq/ngkUYNc+m/7Q";
        sout << "cbr6mdop2NEVsDJbYlA8wPze4/GoRaI9QyzqYUEAqjRnRzpieLRuov52dF4+hxoHqOIDM2yI6N/t";
        sout << "cuf9ujycElcgGstL3lhDUgM7IJrlu8cN9XW04j4gs/U14M253TiIZ/aGLJtXOz5VNIidbRSAL+lP";
        sout << "XcSAgYBh5UYJ7LlDHckclP4NvvIcTKFn/w7/v3n2/SwR5pEUa7QSG/qOZ2VUjj4x3HTnnnjbVCxM";
        sout << "pmJnImX+MaJfmKkJVLlVUqGsYdfUpjUwXMhnDDi+KjniexMhoXREFs6b1k98NoBzh3+a9BQ0039f";
        sout << "pmbFeKymHupB/liyWiRCD3v37ej78/ts3GtcAVYHXolPQvWGJe0RIu7MlWFK5IAN+nTE44lE+saK";
        sout << "aVnr0NWxkYxw7DIhODgP2uqe1R49N/0GtA04kDnQXgDALQc9+QnZRtVTKBRXd9xuV8CasgmIqXvT";
        sout << "LefnlFJAq/NSD5HJ2LCETfYSCpRqDhTreYIwW2/T0kClrB+OBnWGHDyaBhCaJhhXVwAu6FZxLulE";
        sout << "QXMc9mQ/YxHmr3zbXqJ6i6nfFI807ikpWl5QeBwgxx4KvPQpyGvES5/fyEs9O2/PQA33LXXmPa8d";
        sout << "tJQpQ1snDa09cbbWGjAfTmnFEbdZ56zWcc+aqG0OhRwf+mxuKqTpWu2EXFmNboAh5yavDmnvsgyE";
        sout << "RVQxRKhPaQZZ7/JmHQbWIXy+00nUd5Zqgava0M4cbXdsF2r44+9DsWYgRfxj3u76oKqwnJd4unOI";
        sout << "zwTSJDwzIJUdMYSEatWkUV1ZWmHasoHTBBZeOgW/YY3P2n3S4of01ctEBONq6H1xjSfV3+mUYq5g";
        sout << "d/y+gsT0yvYvqRSnOHx2M5dBDWiMundh4n+/Xyzo3dNfp34jYnxdDn0zRxg+OkqYK1+l05AKX6HR";
        sout << "sk7nvUNH0Zuq4w+UfQp2XopYLWVa+BV5/Qm0I1fR+YXc9FE0JHG5pV72h000n1ShPBkm425d3Phg";
        sout << "Mwr4BU6eWzu/sxa4hxv6bO2jPQ7Fy4wY6A77pJX1880S9XyvlGmgI9Tl69OUqCE2xZOw2jdz6WcE";
        sout << "9ttpkay8cktBmhZf1AGfdMEQA32EpK2GYHp96oTHpngieIcLt32CsCoJRUNxbzk+EWbLVEKJADmA";
        sout << "90XtiKaGBbgPpDezzYBVkj00ha7G+by4UpLVnGJ2a0+5B6+CoxafeQIMXbaUBZpQzLUqPv+LTI9T";
        sout << "kX3co0mb4Y9bl/9AGRZ+Bi2C1THYqVogayFEddR2AS5dOX0I7C/XJiO0pvPYZGSbAEr+ONq0FLer";
        sout << "CM2wwvk2BXPuJK9iY6PZYwsmslpayn4+nm28zt5WMww9ZEYgWQf1iHS17aEwOSTukYYT8gZkbSoA";
        sout << "kqMWWeG3GlXiw347fyca1fezHSEQewzQQMPZhfaUZM0zUxXOaHJncWtuLta3zvqeNil0Ug9f1Vv0";
        sout << "op19HAucFXMfPhVEJvQMuJsPGTBsimvJvRFpAGMwnpkke5andXFRjmONELmKjdR+sdGA4mY+z04R";
        sout << "YTYSJuwvNlaFrZ2ksXhwJ2fLKbW6Bm7olAaQJyUH0UlW4pn+Z7WF52DU6Yuz3/MEqNaF7A5yc6Vl";
        sout << "hgjDWJr3/+KqR+/N9Zj2HJc/0CRnBxXpMjSP9lqS9Gej8J9TdGNmgRIDypOv9tQhmZ8Bu0odSrqP";
        sout << "1efptbqz8WtaKn1Ma3LfNnpWDgbJzyNw0dtgPAZPy3Qurb+WOHiDcMNMbQb+/+foFfWLBYqOZCE/";
        sout << "QrukeZG3TmOZKjpmsjK93bJTE6XZ0aSHDMAW0Q7Zvjl0JH0ci/7KwFJSupD1YUEbcaazcDM7z2Ws";
        sout << "JXWJcRNyPJXn/kc2s719ymtmNN6FxWV5u2ZA3SlHEqctEr1WWLGfnsQtWsDO33tpHLRhTyjt9VQF";
        sout << "fOBvem0DxP10O07v00H2ON5BgY21vijM5tYkYIlqs1zABUkgUhlc30xvWFBpD5B++vz6WA9+vR2N";
        sout << "c7H+IRAPHpl9yvbRWFsoen0rQ/oaFCL6ZsiAxGPmkf0q2xbyg2dQx5Ip8RrzX8Sf+E5RwvcRD4Ic";
        sout << "uXCqTO+5ilSbw10biq0V+p8Vfsq6mnjEcDChS23azSaeFgwq9bSkKtgOYb6+gIRQdiwA+MsSpmow";
        sout << "P6vtj0cMuZtdo1PB5i8Ubq/myfuz+vAQEnFEH8DUWn1quBCTBJ1eThEzslablG8H5Afx46CFpP1E";
        sout << "vKr6ROaM90IU423k1jmDRKCUn3vUaqxdqjZFE2v+0cV9nsRaGH9XV1mVL66DBwhKmJKy7zAnKOw6";
        sout << "2D9qM1KHLhJwKLqODaQfMgZClidWe+YcVqWHjwsPX1EBHhXKilSmvINKxv0XrpeHqSBIBLjITcdl";
        sout << "f/te7PGFkHSx56W8FC9PwMuMFScuAzn6cw9BcSvUqd8F+rJ8sliF4xrPpYfMZrYIswNfHg12XBwD";
        sout << "lY6CqKAMLbDaryCpDgDxEM6y3jtqhj/3Bc9NouJaXwUlYtBqMsfSpdYcwNiePEQGdToQbiFyYuqN";
        sout << "Fl5k6Mhf1WBBHOLpQ0LsM1Y6Dco7uO1uclIcRiBy/62EH0PQAE5zA5p9GCdf9SBybLR29RCbHNui";
        sout << "BskcnFFhHfnO6c8ymYcGakXNO2Zx+0NMyoDDcXyCLIRkD16Vt8XcWJj0/JRWdgyMbBtTfjVmB/eK";
        sout << "Elwo3XT+PyHE1gdHEgZdMPO1+6EJIyM1u9WK8MCd1+P4ZFYVrbuZxq03y5GI4oxAaUsgw4OSr/kO";
        sout << "28vnKL0DI3sfZaoSoJGTp4S5wint3rkvffv5mI75T8wV+yollAHAdA/1Ls1JhydDACPAVUGk169c";
        sout << "pj5byLLykC4KZslFr1ZE84eZQ/8HwgfM5Xyv/z8JCuBM3zRxpqsbysY4IObK5vDRCq4UwYzScPeA";
        sout << "IBZYpU0GI+8KsvgOMzTnsdBplC7g9VlyDU2hOTDjwnH30tM2//+TxEFdePOK7Ju5re3GIbiMFlJ7";
        sout << "ltaBQ+WilYIo5L+0yYdqi6O/IEDGMeo6c+/nrb3D8v/E0b4Vsseil0h1/IERIchDD/1jOLD5xDpK";
        sout << "wtIVFqXohw653vvReO5CYOStOKwK1NRj+6evMwnYY0ANcuHvkVcNrpLKuldMZ9wy14/ARgiIjAVH";
        sout << "0o+/xo2gFB3b2r9UhcZR2h+6dHClga7YggFuGxMM0oQ5XYaswSyHhGmRswU6VniDDqqrnbSHz6Kv";
        sout << "wjZI490tKj7F48/ovOKPBXFGcAoabYyh8ItOqbjrM6feVgUdLMEj2OtTDzy2pW5n/G9JPOxO6yxh";
        sout << "I+O8KdTVb7aeDdh5GLoU2EHAggWLw24e3+JxInqj8d8AHrQScmHDa3mAY5tm2+PcmpZP5JpQDV3s";
        sout << "QwbMGaZuVcDIUTbvaDZ+RbBjHBCK5mxM8+TCx7j4v5iLAcC2e05sUtOFIT0QLihYRmHCcjuP5AWm";
        sout << "0J3JPYQPbqMlKSBGRUpo66cPx/IIWB9g9rclpMeaXBx3M2dzqM+5JvtyTf0CPvoqaaQa3R5cmJxp";
        sout << "xkSJehYgagfNrCOTriUYiyiZQMWQ+n6Mo4yf7dgl3ARmFKCb5xWG8Phaxw/clcVlSVMfa2Jm0Ri1";
        sout << "XjlxCe4aZGcqrpoHJTndwDbX9U8YzyDBosTH1+bLzdHzYan74R8BUwAyleswep/5QF8pxYDdqu/t";
        sout << "/iqrnfFHkid1fydF3a1AIEMJdyHBg3JOjrCXVQD6Ifj5ow3SCGmAoI2pkpBNy28XwEiNzSFyoIAM";
        sout << "hs5/OMC1BjzYHb9y2ibIHG0UgWAsXVHFdgz+OiQkk/YtnGB/GVfps3eFo/8YXNxtGaSWQutk7kv9";
        sout << "9d5YrbxGgVNOeB5GQw+36hg+Ud+pVhm4rj+d+c9hiUakDYpZj2yD3j8wIcCpykjF3Eb3SSa0fmcH";
        sout << "CXmIMkurYNTvgcv+pqSk0I0NC1I2CQcgf6xEhTNEX1JLFAbZ/u0bVJZdlrJD8CJRNNoNeycQyQC1";
        sout << "MVJEdyVcxsZKWpE5CwkqnAIWzdAWUt3bMVPT2ofFcmia7iDoOl5oMGclqwsC4pbQK9sDqpkiPTbE";
        sout << "cJtyJh+kqkm3JF1br5UkEnxgK65yWVTu/4PBDax6uYJoZb0tB1ZJnRPxfE0f3KsxiGv6m0S5vx5M";
        sout << "FblX7Lhuoowc3dmkN1RkNr+yFF714V8o6hhxDxLvi91HfS5lUPogCFF8dhMT1xxvWBREsq0qf021";
        sout << "go40uC3E0mlhDdcffwR1m73QFHlsaOgh3soPT9CLP95RGBtvBWscuNjsP5dx7xDP4Pajc9grU+l3";
        sout << "IC6LreGtTj+rXwpEkWPT2AlgkOcFWObnvpU4sWEf0xi7KI90HYseb3ceelpFOr0LcHvrSDqsZabZ";
        sout << "zIX9amNrSB26lf1F6nphID/ExB8E8bR8HDg05/OxUZWGKIT0DaocTY/yCd1HkhosPwiIBIMHzaYy";
        sout << "wLiLZIgNIZqYCf1yM5iVU6M5+Zl8JL4w4DBSo5z78bY3HpdG8u62Z7U46nWx3UXnGbUeoCddD7mM";
        sout << "lvzmbvRWwLfMBRoWjbTYOh/YIjVzhF4tjxCvCMoOm/SJulAYEAomP+oDS6ou6zGq+2ZybTHuQOu/";
        sout << "LAKzuZJL1Gn7tYSBegHsWX+mE/b2xniWd9N5cUyPFOvE51/zHAmIP2WDpFwg9liCl63d1jhGCEca";
        sout << "Njai7311nCZPr8Vwyrpo+tHhzIj77/BKk4Y/bAskdlADFVv/L4jODahPROoVSAQIMjHsymmVXva7";
        sout << "wO+72pifn9q/Tx/1R/namABkKwC64MSYv/a0KrhWwWFDnJAKotAHzDkADxv1cdA0WjaIAoisS+TK";
        sout << "dpMuwobjFIv4v+KDGLCX/qrgPcvc1chxvnRwFw8ekCNhH19Tc7ST47+TjuPuX/ECUSDqnzls1xZr";
        sout << "v/jLcGGT2NuDsBPNaeskj8Yrl1cqqseZqtUSwKp+9nebk7uwMihy4yENvKznPfF9Rd4phaMNzuZX";
        sout << "1jhVr2RUTrQf2cr5QHOIVhlX9BQyclOXByIghKEsijKM8HAg1qpRF1zVcI81OmqEU8nLnlJOhDc0";
        sout << "bbVZ3k53Is93wgY9WNTbKZU2UBuZIjIrQAo2mwLRPuGFwDXD4ou3hMdG40vKHg3beTdQa90YANzA";
        sout << "lECeqpnahX3tcJKhQFBNsE+NLHO8rN84GywOBz/v76NvlPop0CN9Ik/Qp+CI/JRSl++uwfQQHJRZ";
        sout << "3rOhCTPuEY8sEKYzyYJgdVBYVzU4530KYAT5cWeMuxgnVJBMX8I+KD5fAY1r30zGKTP7fbmeIdBD";
        sout << "Bsp2F2slUXQzaljMa/9Ut4VWi6EnB5ZDclSCl0I2rPfUjljmqEeoI0RrRPvx2gvI/s8rBF/RVm5T";
        sout << "jAkepflhoxBwhcGmW0DjbJD5kK/hGl8jwq9fJUkEg3zw8tKC9eeMFdd93l2toj8fN26mZxZ026sq";
        sout << "q/IrYfUu2Qi56HEdhZUL2x5b+ZMf08KzGEaAF5m4CpYZvjDkMZe+nOiBf1RmqskGDwwTf+a7THnO";
        sout << "VMDdo9uQc8Ubg46PjX1xay1zRS7iIQZgmTGEvxaUlV+uTxS1eo0VX4H8SgNT3cuSCrmcZ4Vb7vsK";
        sout << "f/aPoS4LYaSOss6H9mcYd7xO/qs+2x982O8JlHoD08klJArDlzkABWL2pHvzvHjCTkSI6TUtVmAJ";
        sout << "Aj7LnmOJy8bSn5pK8UCkWFHHCjg4dwF9F4dTrsXNQK062sXDvpRQIQp+YoxRHSTgveyJMCfv4jM6";
        sout << "NTme8z+4Jbx9GY/COeSO87t6N2ml+mkAhRHSq4oJL4TZxlwI3h7YezURylRb0IOWMwD3B68oP0p9";
        sout << "lc7W5Lmz7AGF+8fFTv/I9QjUm0sZAjrInS4iDUY++XdemIl/1QgOLWW6UN0LyUU2f9eEY895E2Rw";
        sout << "AWDKeuk8ryQnsDU0+fdycqL2kejv3YIGuvz3UcpboSQA/zu77HuUA71CUSFz69CMJ5yjAAkCIGna";
        sout << "cO9u7aP74V9ZdTZ9J71tRAnWhtMXg21DU72mkdp335o1Q8Ckhs76ebwsPtaNZwFu4QaW7NsMCd/U";
        sout << "80eHjnI11dsUbQN7MAAHZHVhl1f5BzGXEMFrivL6ZZeKiMNpZG82D0HfKTiKwx5E+5Pv26MtYYzG";
        sout << "VKuah/nBTKuUOeoVIgpiqZRyWior/D+72mMWfsZRjuv95NgLY9EFbzg/L6KxmhQn71//upROMiX6";
        sout << "DgyP0/qPDEW8TdCAJhqJ83hGXV2fwOfeLh19K3juxl0ifK1hYyT2da9WZAG1ppTPEZM6Sdo8ZVBp";
        sout << "CMZzrFVEps66Y213nnk/zDYU0q8uUQBItVJ8aFEiQrvUE6YYM+N2UT6x0I8YRFStnbBsbbyafQXs";
        sout << "OBtj+M+rvUZ4mks745WsK/2Sz+fhrHfzkfF1tT8mPGMp8m2Zqhh2w5aNthlogqnfSg5TIecY9WKU";
        sout << "HuxgPMgYDGMbOxvzDisTxnr15eYbiNkNRd59xC0D8dkaBFZITiyUxZtMnrMS4yshVBkd0BgV8x76";
        sout << "eN7/MiZjaj5NL+ZyJR/WRmk+d1TO4cpwEBbwDcIYTU/Su8ixFNDMX0/EfZ2CEbuxv/RWrPIA38Sm";
        sout << "KPrxF2Bk/Qqe+r0QMD/Fv/0BGSmgC5ntlpB205YOkUGpU13AzlMiqsvOZ5YPpqPitJV1axaxmS4c";
        sout << "IlUegGlBZVkvPM2nZcMohm8pOWZV4rgBZhyE42Sp3n9WryR2x8Ua4L0NHuffU3MfOT6jTY0tev++";
        sout << "xt6zEhtvhSEdl8y+UWcIBoBtlR8u0HrCZrnBX8MVC5ygnBgjDPDMyqnLqxirr7IhZ9KrLClGGRyP";
        sout << "VK44nIb2OF1MKNjz1ojvtkDsRmcK3HyMDnuW6ueSFGRZ4048tMUSTKWJFmx8c3Rldmxacn1O6nmw";
        sout << "ZdMKvEH9REjRaSTZcRdDt18/TUdEtng1638pEZKraTXNht4VxFKeA7ufziJDjajXN+g6Pjgx9bDz";
        sout << "fOp3JITJIxglbLH9Yf4YhbvEB4SS1H7tOkPeWW+0nFpHY3dinpstjpwFPRV5l6KRLpi1G8K2Q/qP";
        sout << "tJyReb5tOsaHDBP31i0okVmr3KM3GI2Ypk0K6INHQizcWPPdn0ksI0jgJYJ9ek8PxX/6nUsKkoS4";
        sout << "lKdsh62nqKm9V0cRgAlBDf1w9h0SmIYhfOU/WhBJZbzh21ePcbIK9iLjxLmoU0XDmOidRTmH2KB0";
        sout << "hgV4tuH9SmWft5jzS5+Ovso/HIptoy/5JUGrJObOCE3CYB0tRcS94MOzaYUnSchYVBfChzRjddQZ";
        sout << "5UxYm8xa42AvpOQjcAH7+eGDc88j5V3SdY8Hxt+wJdLfaddNp6yHg1eiremfIAf5G+H901BzglMN";
        sout << "icUpcT4aIvNJu6C07Ozz5mjSkg/vTiF3I2lVaY+I2lfN4pSe/hWpzZVqKpUvqtL7wMDYlQbAtaQt";
        sout << "+4t3JRWY9sHGGPSRbbTMygY7xvPJVe+Y7sPcF6E/PV94wO07C502Bxv2SPIrjdgIKsxGQQvark9Y";
        sout << "ukSi3ag2CKflR95svqKbTK0zvw17+T/kq6YKm/0T7U59vXAFUlg+uTPBOiBQxa7mwTRFT/2uf+zE";
        sout << "oCHXzQeDZyd+ggrChOADpskHrO5VmnKZqz6dmavegIKels3myQ2Wy7BBuD/yPlGmd3XDriij3xFx";
        sout << "yiwTMkEzX0e61eVMvSORSv0DVtyYoh0MmEK7mjBlAPhJDR4zHExtzTd20Fe6kV/w1/GkwyehQpWJ";
        sout << "rFBIa1Y8iTkuR5Ij33pGrrd0feQwRoFvZ15QZ7bh2MPfWsnZvte7366zBKzzLln64wd9AaKS6qnP";
        sout << "Cf43VqI8XuyJ+q9goS6LNbcKnuWu2Pc7JtEyACf+eG11CVTN+tAWUOuW8t+8H0M4wQurbByJT2xs";
        sout << "CFhIJvGe7R/py465R+4bLbNHaQEyUjFcFR1LdPL4ke4pBVDdpw8/5es1XkB7h5p+BZ27Wyu5NEQo";
        sout << "+Q0CpHNegbu8VuHfUsFDPxMIW+k760uL2DGXkKBfPfAshKDStZnkYbHip+mgfHSLyH3i0B5+qRWS";
        sout << "ThksOsjI43PFTXfKflPvre/TlNjyN+2O3aNrJ2GQu6Y0Kp8NHp1Luv0CZrhXewnJfaeFQeFlF9Fn";
        sout << "TAVcADFN/60SLHgTbPvZb3GwVYXSxUf+Fn2WwUSNRRsx9/ghHGoTTattwqLigbj9gVkBllSTBTCO";
        sout << "DCc0uVq9rXcDcH21oTE4HOStdLx8EyMTo+6OazBsp9eLqkIybtyUlzGNImr9OnjpBdQ04/Ptyy+m";
        sout << "aXdyPeIZ+MhMPbXDVT9Tt6mNWB1tfPCRqs/3aGV9fxk15Bs+Qg0myJCe5hZU8g8HeGjjQu65/26K";
        sout << "r90Bo7y/Vezu+LU4nLXEWd2mWwM8k5Nbx2hZAG6RJ4PENcIjvnmWw5WXFJ+WabCmFTblNY32cOgg";
        sout << "B9XrQjDqqhDfdUmCD5rNTy3qt48prCL18ichOZZrzaUaB7CFB2vBuZAb5Rn6D4pTEzHA9WnqO67K";
        sout << "EOSYS790whcEw/+DjKsfZqLx6snfV9IT7y6ix57WBCwXiPU4xVDhKWNcgD6X2fCwwgTDnZOeXOUI";
        sout << "UvbFiIj7S3hKxzJwvWJvSeXzXsLGmKmJNneSWoDBeYevJR9hwtnaz387FOnKYtkl8quX8kNog2jG";
        sout << "x8pEYkNMh4nJ7PqxcW4nAPTxpAjiNs6YRyqRNlJ/Hj0uVYoZy6EMPDekw2Mk3jIJQ7oNAJubU9TP";
        sout << "L5pjYb6On6ElmQDYDsqBnI8EiGnaf7PdeWRo7EM4zTpyqEB61HV4m0rjX5oAfOnRLwnAmpvj+BOx";
        sout << "peCUdd4Oy36Ip/wKu5wPNLMb3HYG3Njj3nIotQNEocg6ObZN8t5eE9yExMZqRVxSbb0LUAXV1NOX";
        sout << "MaqIkzSkH/jTPZjfuc4Vo0o0BuByEEOCTeMrDM7JPxrbTVvXyQrYZCr7KJD+Q4kON13gP3mabCVn";
        sout << "LecL8ok7jZ/4jOXdj+fqLyxs6keukNTsIhCySrxidPWFU5PSMrm41NIWwQ61+P/QuYxOpHw8Ufky";
        sout << "W+FM+Gag16MChiQ6CVkteJmMHvAUVr8G2+I92OQmrEBFbGo91kpf0kwZolK5hyB1NlSGY7jkY9g1";
        sout << "i8qTHaJaxvJflhaWKf1CNzk1MMHloFWvKN4wiMhN5FDIyGe3QpC35ntxlL9g7uE2TItxOYHX4XND";
        sout << "yRWEVRfwwXpUVRad5gfnWAecj8m1HhtFRazds9IT0/omdILAA3KoH5Muu5KuIOiZKXJYplaHhpod";
        sout << "tb4W35ebimZhy1qF0Bj0LsTmL6YYCi8l5ISCmVQwMikAcy4iyXYSwrKYcnJmqu0JWIGaU0SkMJ3z";
        sout << "cVhHSJqP7LsDUCBQJ48OuFHhdt8UuYMMM/Q7+XxnjMRjrw/uZKruF5p+uCqeb06NUeC3VYuGjt6E";
        sout << "FDVYgfcut6FOKP2kJ9/ZLWuSTZAQrjFtPXwLrFLx3dL51kGEF9uVftdvVPYCrO3qrcRiZZeaEYba";
        sout << "OtpLnvzCMTUaABB15AFZJUGGGIh3zXmJdeSsqhJrLMe5o2CyZSifej5Xnj4HNIAyNsi4ftESemUT";
        sout << "ulf12cxOr9P+ERzj7EFERdrZZRyDA4KI++V6pQyjnVfTXHXGk2jvKdKs6vI6X2qNgHHCT6ww+VEZ";
        sout << "uPtTweDKmy2tKN+Li5Bh2+cNQhc5DQcpW9D4aJnl0huPf9HRv+P2rajsSbVyXhFcai5QC0hT4QWg";
        sout << "I4UArbXrmTjJ7ez53kg5uwzA89EVZgG7QnE7VA+4n3lvfcENy4i2faqYH52jsYxtEFCOeOQ0dXs9";
        sout << "kKubELGvFgkQiLfhpD8RarTznN+++BlRNEXNUg0x6/1WUUYsRIdZjGppQWZmLVRxBrfDnizxyfn0";
        sout << "84NmRtA1BjQtZbG/gRggIJlsV1B83wj09V6spystQK41QZKrB9MyEbAKy89LI93uJ6Lde3SPimoL";
        sout << "ydyZwePNAQbBMc7C7P+O+MH+nOsOxhBmHgC+85OrS2oy89RoSfM1GXsmAgGKz5wUomQ62vbPEP9K";
        sout << "UZca6T3OFMbeullfy994juN8uTy1CaoWyucFSc2eyWKD57WvuVPbf8KG02d12+Tt67SGK/Qv8gnN";
        sout << "8Mb/5oTyW7sj1FVxsxDt2bxqJ2SjkzMcIsHD06qefGPYUG6pGm/fWa8wiSqnh9Yi071mDDCizBS3";
        sout << "/LjsZjEX14cdMzSXLQKF3U82ayY4Kt5/ufFs3BjrLTyZAHgB6WyTlxKMgVUz2XxAHM1yUVnyaR9c";
        sout << "HKacov8+Br3g8B/wOXjKlBvhKKh54Rnbzn7e8/lFXpc5HfZbYD1GV36QnyuvNNmvDRBnDrDnzLGB";
        sout << "KqwDcOYWvXnQI9cgAbhlHaaIGpqJTDhueSppEapR18hJ0asiXWZ5Wdn03b76Ih7HC96AWDLbghn0";
        sout << "pSb5pAYeZ3+MR8x6Jm0M5xA9qNo7efoPBg9q8TCCSp9xSAHp0pKe459eiq3xPgUQasMLs/TYqdrW";
        sout << "FWKP5E5pW2QvnE++0JCMPl8rrIDriGh9wuw188wZBYw9KTKEduhBljcKeQBfk7Y2PFFWFgXpMnlN";
        sout << "9/3/yftoG/eWihRXMMd78n9575p3zGMyIWUFk3yIc14rYkcITF7YYlzA9Qz0anNgpCVUOn7gqPEh";
        sout << "xmgx6/Dvyj6rnPaV9vinZCJKfbkMhUGyTzIOmNgAM+XL4YqWdJwNYb88Rj71FNkYqdlrIMElhXgc";
        sout << "zGfk4Z/BIVm1vvU3NuCq2HxB+Dllflc3ZtB38zhOT/NEsOpYgrm1XDWAuhQuOJrAD/dJZC+o8Vun";
        sout << "TrmaayDOTXLb1w7+fRorbKkeS2MvDB12Go2dAGoqnzTOgwZVyMukhiShYs6wp5A6qtWnkIZgK4cs";
        sout << "xrO49Ts+CYkrWt4wQKQ7tFIUU8tbNtBTY239zeSZ/MQdGhyBYCi56vuR68qUE+MkSgAn+dNK1QtH";
        sout << "S8g8k44NAsMZQ9YxvmrPEBYb2SQYpK0zKvqtbtNTI8c0EjR4FDSehmhafVd52PadOqFho0THDbCv";
        sout << "z3UmFzvykQUJrMbSbww63wQ6j846Hb1yKP24csPWb2WO8NEUZgSQdprNgVg9VZ/9w8yUq4sXkpXt";
        sout << "54qMTC8odbFpaiuy+ezU+83JMK6BAbsaa4T42nqMqp3FUrxHum8E/Aq6Gci3pTZFw12s3ocX7xMb";
        sout << "dPoDvrnpThnuyKP/e4GBY1QxNpX/wg0dkwwW0RWrv3iGsDCDZfWoPOxaJz4E0HxPyu6E+ArIAJIc";
        sout << "/KLA1FpMPqCk7U+WTYNRfgKTM29wcIcqjV2gPqkZl/7/1Pk4IgT3kfr70N5FBYE+poV2l/zZiDwK";
        sout << "bWI6c2q5Nthj/V2F4Sbq/f+sNnFtPDqbni16NopBAlSMUB/7gyR1vB257/5EHEXAnZJV8HlMbc4w";
        sout << "jDp24vqkOzjc926ql1dguH+RD0LFO0DZuJZUniT/Qrk7yYjuftuVvXhRWQHfsKeiBRKmYBVekBP0";
        sout << "v2oWC9BcpeyUEoexXMb7nhYxYaee8hT2S8ExU2uVkTrkVlyJQX6RppoggC6068Wkvpwub8la5j8A";
        sout << "GWag6pGQkebQ/vfj0dLARbIGkURcXYzpEu2iDH3LRkZA18nR3YvnZ7Ds/N3lExjy+IxbKh8wvV83";
        sout << "OWgHK3fcJ/aFePPKv7nFoHmpYwxXKJsF+ZrnYstS848Jklk2ehxd/FeP/4/WzzOytmT/wAPCMzhB";
        sout << "ocyZJGS2FnbQ9r9cz+vl5mPUSZm2Hc23WQnHDghGSS6ZsVjsfrMtlEvFRxIyrvlUJwvnL+g5GYYI";
        sout << "q4NYtTx66WReERmgL2hipohEDbmnFkX+YqfXYzw95LDFM/UUlzt2F2o46v0LUeEqNskx2JZdBJUk";
        sout << "OZjhQDChnF98lpzt1hB8OaCr9IlJ8xfaYw7Mt5xdVkVd2HGRVzTvgJbtbXsJkO4DoLfXYiYFeX1D";
        sout << "Of9a08WWuhHVXVPUw7JCTXeWibxY4o1osV+nMJiifNQxx9AsJLavtht51gFsXDS58lZt1vvw010f";
        sout << "0RIAfQd3wh3k63wO/JRFOjjuTxVBAdSH+E1KYfHSmq5ExB5bAjTSaJeWOgYPjl9hfiRa0Aq4XveK";
        sout << "fTGm56vdjpYP0mNlgkLODtZt7FiWiWE2NpeixIeU7twbu2qk8BPFWMLg1ZprRBlLs3P+The4OinC";
        sout << "K6WwbxzjBHY748PpIYzWtemrd23xNQnF2vtXeHiV+BwDcrdJY0pG+BK9uZPI2ADFInfB6brRK2YF";
        sout << "YIyTB17d77/x8zHnjmVfu/JYvEGbpwxQYVjDuRs1UvhlpGcQggpx9dwzkfNhTmdCzKHDxjKsxj6f";
        sout << "8uWyaI0p9PNU5jLC09MeMWcr604VoZgbxwOx6DABmUXzOPYxx8UTytHVCdnvZYYlEgulFdqCUsPR";
        sout << "p0wFcmeHiu9Ylfr4oPQkn4rhWH6ZdYsMjXYcUThEOm2dXbLAZ4tFVvh9obDPaMXFDKQoFqXHf0V1";
        sout << "82DQKwM77cZ47uXmktetrq3okftL16MlsMjxuf5oieezfYimSZIiTXbEXjTqoaw6RO7KTvw4BeQW";
        sout << "MI7xGMtopI45SZ8CJ95fZyTfoS/GywmdMW32Tv64pIvUBwfVpmlwdi7u6cG/HMTF/8hyVr66XEyO";
        sout << "/qCJpmgfYSvmPzCRI9k953SkuCCBcnPTEnAr2kYePBrYC8MUYZC4WgoNY4pTTaOkqU/qgFnxMPsC";
        sout << "pP77d3Eg5IjPfUwHKkqB2WI9aOwAxnuMu5AP16phx8zkBFn8mYAfQzo2eCf8viITBWUGxjpgiE0w";
        sout << "bj8rCX7inP1thyf1iMawvhcyq+nEM5EV52+2SM0BPhhNP2BUcq4XxMITRo54ckluIIqVd2HFRnom";
        sout << "o4FbfvSWyZEXL3loWKWsv7YnWxZTRDlC7TwO/OhxZR3JadpR9q5w+mfQazCIpn73Bc2/6fWBJbi3";
        sout << "Q2/Wfm42d4CpuxpQYLTBonMb7tG8fv2HoQgmhVRwV/edGx17sgteeT/iJmG9OsTSh+b4Ts5sfQqE";
        sout << "839h1o4eO9kqHDzeR82Bb4vpejlrtJU9+PHIth4FGuYNzuHdRPpy2S+NtpeAf5L58hCa4rEfxWfn";
        sout << "KN9C3+vZdsztGi/sKXG/NBw5kmSJcTwyROae68ehmp8ID9zCYtkM6Lng0y3l2GLEFbnwd1TD5hFh";
        sout << "e+wFJZgD5y3fnZRiB4qSTiuAZ6WRUvTVD/jNLvzMzP35SpmJkap5dZVvq3kzM/gMd99wyWBwfymw";
        sout << "Fn5QwRUXoNNgmWFs1ljkbtZjYGYFs7FzRm/sGsmnOSbRA5bMSXgGhnHvUZhlKQnyhHNT2e4CIjSY";
        sout << "cB0992CVhOS0QcXGltKQKUWUXuIAMTXrii/7sZPEAlhD7DoHkc6h0ejlGDBn5MQCgRNUfRSFDrSk";
        sout << "BBb38mgjSNm6wIr/zM2wGuwT5ziUTLPjRDiJaEwyfatF/tKU4Kuvhy8PF2639B7iUE6eiLJmRiX9";
        sout << "14KFSXPOJ+cvqyhnlKIscELEupsb+j8OsHF4y0DXfkmi4F0mrV2jS76DxmNTOLxy6+nG8ibl1bAn";
        sout << "1Imcjkk+ZH68bPitSnNiKR+CIwnfa1/DeGpkYpdY/HR3RA11zTsZHAEFqH0Rd0r7Cp/0gjhnR7KT";
        sout << "/gfMuAau6U8MrbqAK2PN5YUXh9RjBh30prKEfHESO4bkYbhJ7/wjVggqVRIl9qAu3FBIGpy2OFVQ";
        sout << "KP+RFg4efoOQtr/qjmE6TGgyRwdeSVTH+kJWjJm60k5pyFdPhK8cACe8fJWUDePQ376TPB1lBKiT";
        sout << "9HvENnbFK7W4T6wxLQLOD/V14frVEamqFBXxegHSN4VB5zrmQ0XHGZ8esevPyhZ3BuYU/5f3Kmbt";
        sout << "kmqya04jKXQfu0zwlxkg2sYPLkzdoNrtQXwWxswIMiDOozE90Pt0iRJcVwGDB/QPaEttbLscfQDX";
        sout << "tYuNZFRXgj021vnsSxvFLfGRHwIaSpF79VyvcalJ/8BkD5uBE5NJYHWcCvYrfb/kDkKj+1kJaPUj";
        sout << "OGCjw6FcGrEWK903hXwlfW++30CO7r0IqbYklwP42QzacZhv8bvCrU//jkYSAIOYJGl8//9zvjdo";
        sout << "mg7V1hdg2Z1u6GsMpiLBGuBASjWWYe5d4+lHestT+l6vCj1VGX3RwIbacO/IY/nsiWMzkYRbTpXs";
        sout << "ga88ELjA0IdVgMqYawu8zH0MWtDIiLvPmtEjpXs1eRy+tEG/CWJI+hPE/4//DgDklykXiQfgHWot";
        sout << "xPPionJsqNdY9qlIjma3I7K8IkV6AHpEhGmdr07KOMuEGW7rOCIEVCTTdA/EHdOrmb0sGWZZTVvN";
        sout << "zZvX4iXzzGZrNkMh2MvcQKxZ1N9o2ujRMlM99rO8dJm3Q16Ma1rCT/7XiyrFo8Y/7rpCyvBeUsu6";
        sout << "qGIlbBdS2+Mx+Jr2YOa19pRs0Ps0PxN3HZ2J/qZJ6nEbFLmublyaFPvIi3860EA9KAWEjSEzn7Gb";
        sout << "z5TEw9TjisLVdAZKlqCqlGVYl3ttekzzrbbGgkq9mQuLeNv7BBYyigf/akoQGQ1nMU/U6/FvLVYQ";
        sout << "pTXTNi/Agsy3SqWo8I5pRtvkghn/Q1VP5C1UeqbNgk2hAPQ+KINHSZhq/WrkkfjjBmHI5FefH5Hd";
        sout << "wGYrHoFSpsiRITLN66A56d/8a3VUNHIJbBS4M0Y+weO7wsj9itTRR+A6wvFCM5vOaE1B3682QgCT";
        sout << "DSXf0+gHpe4L3X5dsaF6yWOIuL5fYc91p9dKC5voNaZHX6YX4pdfG+JsLn831FJUWQBQ5u3mlg5M";
        sout << "VaH6ZitjbAePbJGt7RY9j4hCq+8AljaMJmYnJWdLVFBIMEDkC2ns0MkrtAJXdeVYg7pw0CPPq2KU";
        sout << "LGixmuOwLrJh4xG8pwf/+VAdDk1HqLUUP2PF0Tt5zLHV9ZqFXfv8BjC30WcIM/FtWl9Mhl1JPXAB";
        sout << "LNiG7lMoragaWKNIkpIQzjes0sGDjOKV/+Lc1xLlEd9S8U8D1w/dRx7XK8/7uEbUsp9uo9nAaneI";
        sout << "aAzEE3sFCUszszrBurzBJkHxMJuOpRLXWBA9USND13j3J77tXlBkUZgJ11im3fblE0C1kHFRx2wL";
        sout << "mdNmoVhysYAG/0eM7fNH5R0J0XwVbizvB9V58HF9nqO7glP7soT/hgQv6uZoKHog6F/nwxxE+1+l";
        sout << "G6fY7mh7eQw4qe9/y6ZB5o8J2mVBtKa+QdAr6O5SmVsg5G0X9UmDrPRXnM2w+AgS+PtULtmrZEeC";
        sout << "7pk+DWZpfjwVBBOYKb8KVGsj1jeSL+PtJOjNOIOTBSS1Cx5a+0zpgL6HXaZmPhE8Dey72j05TZMK";
        sout << "FdARb2U2AsRrW7Ri1EGpPPtYfAV7x73oTTWBKJzy4sPapnieK9E7oDq7lN0kArS7df33XbbkO2l/";
        sout << "Q0EDMuZgT0pwKQOKabaxmIUDDIp0CYAo0B0B8ACTQAE4gqbRf0/xDdIBIxKF7ZTSrAHOIe+QXiJF";
        sout << "ZmrtCT+Q/S2vZX8yd2o9yGFrkA0fDKMs9Fygd5xZJYwWsYEVPrGOGNhFHmjH9bl/zjSb8evWieYM";
        sout << "DLnCgJ0yRb4vWs7uHX9/nlWkT6l5qA8sEzGvL+NVtA7Ka0gy+LSMrtGjWzqYQ0kua57m6fKYhWwC";
        sout << "gD3USfXdApLbs1kgJ6LhI40c7a3pSdw4EK7nhRj+nngFPVZNhE/t3mEb2DMNvI1EDghYCMktxdKL";
        sout << "60Zp9TkZIU7evSVf0Bukwur1t8AeHjQQ4BYhMaMaG26DhTQjyTfB8gvU/IYL2YTxOgFw3vqTWfO8";
        sout << "FLnmS5TQzZMAvAKPt6nyYPXRPqPVlywiXrc49sbozaliJooqiHPySu5rq1lkJNRRv6rzHddkJgFv";
        sout << "/hS2lMx1X3sW9vqobf2XAwSeewiiD6mvesVw5VCLfOWCioWDMw2bMk2MT0E2YwMjdkM/Jg240JwA";
        sout << "IsYWWfXTBJoU08SNPlUAnMUl5z9wvbKvjTiZgqZpePmWUW9V4YZaoK7p+JM//VrX76xpG7k+C+/O";
        sout << "kmKzwUlW8X9yhwIJQOir3KIMlOK3zJuE5qvWzFMAtpZ1xyaZgnezaEmpQdVlzLisTJ9p3rQsGRw5";
        sout << "vpaNo9X4PBViQ1v8VncrmJP0DMIff1iVXl8FRehBgN8rTtELREi/Dkscu77Mpb0g7U7YoEqELPq2";
        sout << "PmXXZzFduFMUioA2AYH5GTQcqQ+btk2VP07XQDd0ssbCLW+pYq4mSoRnKidxgT4zy1yzyUFOOo+3";
        sout << "PD32GA4Tz9WngH/WDWMXu1VzsGmAPVdixVwfInjzWZ2nUCrUCEGxXmdXdZ/ZOWzbqpUGjVUqBc79";
        sout << "XuiulTVxE71nw9gVjS/JtuTuC0tV/144LRHtFCm7uXrgeqZA3M3Nnxn/Ta1lknrpe8Z8MrSRU83q";
        sout << "ZHlsJRzy7RFgR9tzRop2ehbMb9KK2clK7tFqFs3cIuHVTM79JrJRNoXnbMB7pyZT1QpQnsVU8ksR";
        sout << "zGhg1qYXOPJL0Dw2kUMSe0YkRSAM3458HSIbfbgkAeDa25XCkSEElKk9e4NCDZ25sHPLFcS45MM5";
        sout << "wDqOz1uFqXGbIKJdLTGrN4ZfSP9r/Lsl/ZF0Jq9ngvdW4gCU+/SomkZWEOFLAFLquBbsVsxgKruO";
        sout << "PCfVJgSmKG31Zc3ejqz5obIsO2BtzrUEmVPDvvT3aKDbUvg8IB/ktv+YpdgEzlIC/S1dsOEcahVP";
        sout << "DdyAercWcjnfcuYpwRlqw+s3pG/U4hdcSidE5jYPOpv2L+sDnXa9iWTThNAEsIn9ZCQCyERGYJHw";
        sout << "IE0aONTi9HTu8j341qRDYesfKVqrTUnbrJOKpjHdKQtCoJra8eL5cHq1mH+766KEjy5eePIs5TmE";
        sout << "wNRwH4qfd116+/Gq5n7Jdb+P1pxuzwaBfQpJZhR2Y5oBGvle+EG4ryDpUN0r0FM66Q+wa261D/RU";
        sout << "yidWxQs6kg83u6EhblsatBFio+rfI5CuWR8waXmAd5ZmLtk/7WLlEFtAlhdvrXJ0/xPPC2En3Kg1";
        sout << "Em96hgXLcdcj2lJX9/nUhXaSsyO4LJYWUFGkh0pSURoTz0Fxxva8HhGfvlSgOtZBnn0A1kkm0CqA";
        sout << "zHWBLfQqf3lqCZnXer14bwZMGl3vxo52Zrl0TJ+cXbxnBpgnuil8J0QAtdVZAedLaCgpwzZvty03";
        sout << "qaetKQMOiIeRDgsp5tyZXNKv5KSFAqjzr3qag7WzMUg/R4MeE5vW+HmHW0sl7r1H8SIEszAZ1Ig4";
        sout << "uifgLXq8/DWPUu+fKE+WZgBNQNtqFzy8/Ak+lGx9liAvB3rWT9FikSP/FJI3M68xx37o4SzTPRha";
        sout << "GSUK6N2TIqOs7ABHF+MG8vStSRHk/ldLc028NnpArCEk4TI9VXnYES9EmcEHuikeFduDmUuPEq2S";
        sout << "5VuXIRsOukjpvA8RylhSwmlo3AHfsvbooF+ubaUvBKVuNnAk7veHRk6TElZtMgkUE6JDzzQfQKsu";
        sout << "0vL3rZf3PNh+eGMFH6ddvWay5QjgQohLlD9ayQYmleCTphD4raQ0E/kHrfZgUJUx//cK/zNATiPX";
        sout << "71bDWMzocigIsH53gFB7uPTyPyuqjzaZ4uXEQJZegRNt3TCQ7HVmZ4agsre0IZnYNo7Q9QuAhUMG";
        sout << "GsNaK04TKz6d8nawoqgW1uzkhuBdLSLELYUIQ7hTl3YpQtbt82RlLgPh2PySWhFj0jmxciS13yD5";
        sout << "dBxnRwjG8pZfutYp/BnxM5hGZCA29Ijg/zDSbsk6SHoIMnVUScxBK9ZbJ05Xc30VyD65b/H52A2L";
        sout << "Fg/3+qlIAAqgwAFYzAz+6KLifYPnd29WEgwQekCefzpboSau9ByhwpQz33R1m26AIbc9Lshth6+I";
        sout << "fu0RGwtCqq5JcwiqIlDNVOMn4QvocrlyDXKnby2evQU85PiU09J5/uYesfWxUQhGcWNfv/XphDgo";
        sout << "xD3WLC17Ph54GJY7ySCnKrxp8aydkRo0tqbCtdrZaafZrvxJI0kAUEGrlAgRSTfDJ6DzrySwjGK3";
        sout << "+tFuYIfvDXXBTWIoLM+sE7FcylNgu7phkSccrHuYubPzOAGzszaYrRWN/17czi+1kfs0VP8p4o7Z";
        sout << "1rFAM5GTw3IE+tVOW3UX59BIIPh7JK/pxj+Wm6jgBW5nmbhn56XQyhWLIKGCTWTbFNS0wZrjqfWu";
        sout << "qCcsU1xa3fOk+cXsPwQ8QxDEDDgBUH+LX8YGQn9aj7nPZPSzMyKuDStKZ8bXom7SpSJaJ8QTXxt1";
        sout << "ZYg8TSYj7lRtraKYN0+b2WTPWwkgLVrbcMx/HtpuWmsr9n9tp6hEthhrgecGD8oSc/DZsuRm0Yos";
        sout << "Durv2h4t7nM51z9CfbNM+TkyBWsoZbyiE0r8IxY8p+70VpXJnmd/z5lqi58aYssu30FbbaoVMwCO";
        sout << "s1aOjj8GQ7u42oo14zhaGr+ZP4d24Llxb6OZTqq8zE69bYoHyRspqkrdX1thlMrCU8Uq8cUBG2tq";
        sout << "8vr9j57Xr3FCFxYtQPyzfJJcg3L3zSCFnKmj9xgRaAv3BEGy/cYMBu+dXHe8eLOqcQHI9BhrkwTU";
        sout << "dctvW0TQmCbC8bsL+jTeOePvceasnObdIYyFM5IlkJpe/jg4QysMTlI5xnaGNW2BluF7CfImErif";
        sout << "z6AAPBm52Igtmg4jVv7P+ftyYpp7IJwiJM7nzoGbiImmmIsgtV2sN5V1TLOiVEpj/BfSSiOMxRnk";
        sout << "QnblUQb5qnbKZPSjf/7qpAXQN8r2pDNZSWd/oPomj+4BQ8HTB2EeTAIiwSWPO02CibfYvNEDp58U";
        sout << "j0LxCwZmlykskwYIqnzNH97iXlZrv9LFqhEM7m5QwkCB9FPrzm7RancFhIdzlkLbPyUUzvvZKACc";
        sout << "Z9pbSvvvgbYRLxJmUHoQDp57i7noo33USlcFMtQKNZlDEyRi6eJcsR2tnyqi+e0B3+guUyPAklVS";
        sout << "fo9R/7QJ2+tL9SEQ38kFWG5Hf6AjFLm3AhXjZ9XKZfmP/nn70khHZWtOa/G5bDmu1hzZ5Rl8Kkcf";
        sout << "48Wu8Y2xPPCbqa02xDmzysctF3fAkrfvoOQsgjIrDAYDJaDL+DuNwg6EdcmJiRzgl2akaxH+K2iB";
        sout << "L2rRJw2z9cFCV+eYIGzyd2qysgJNsTkh93tpmizF68OlsLx/vr1lZ/xHhRk3kzodnC1JZJVXV5o5";
        sout << "54fbi0tq9I6F8sgnsg6KreDXIG06L9bZniDD1W2e/asaAC2U/tsp4BpoitsAjCmD+HDOW0PlO1CP";
        sout << "HLDjwzY2bQ0P7x2+voSBNxoplwRATdG/0OXh0xB8Pdae608P340BrZfS3fC/5QC4qUeIup3Y+eAh";
        sout << "CMZuq05cXo4hrFuzHNxOSlvF4TcNKpSigAJrFzuxA4P1gFGtlp2VZA63f4AbqemqSvYujSnoZFmf";
        sout << "2KDf1l4yKmE2E0TP42CkN5UB7PGpWi5A9Dcip2AGtudL7ptbrCqrx9M7Y94F+l3FL/vPzf3Jry6A";
        sout << "2Hha+oklMyO4GmfJ5Shv1sU6vvkl8dc9nVJxFey55k5zYgQn/UecExAI68e9G83XNeT9KHNwHc8k";
        sout << "a8txXKspafr5Eydf/64fnSKLlgXIychI7iJ36//adoHwf+AZCvqAn1wFJ9vpFE/Aoa1UMTRJcDCZ";
        sout << "hhWcxWA3ajbbpdWMyfoTsYVU3X3FdMdo4LDoiJPoXfeF+uJu+0uVenoUCHIzfTYRbq7Bkkw/cf1n";
        sout << "0rZHQq7w3k/qcrj1FChhLc4Gb3RnSs25gbElTBp50LmLiDAtg6tEItPGz89rwc1CBVbiFKnTk/T1";
        sout << "4JGriiQt6cquoyY9Z2E8w1UWwQYoIbTQQdyreZbtbHYCGoaxxt0TsX7JH3dTox1QTj/h1LFl4sdR";
        sout << "wlfOjxZYXrn/LTdyIWOyE1dL/ghNi91LVG7nFMFAq9+hIonaTrlngp9TfkALBcoIoasyCMaHAXUt";
        sout << "Wr97gwIDWUAnhHhTvHHQNUJMhbYWlTzY4hWo/hzWP8J6fHWpkv7B/rGEN62lx2LaOnQ6U12Bzzy1";
        sout << "mGRjWAWLWrEQU8hORBSd3Na2cYcpaWrxWvJdsuNOJwkrUIQc4UeFwaZgAAYVTfZBPjVqOJLxZydS";
        sout << "Vq/uvdKXALEi5i99vSMcYv9PfUM/XSb3I894MonUHQkQKLIFGcAfmIEE7ex6zNkEf//cL4AZm8Kb";
        sout << "UJdq214zrzeGfhpvMZHxiyUcrYefVKJD4yywBqXyU9K6SqehUtlLK5bvCdMo4xHlS0BJILbKtRxi";
        sout << "64w1FFYYJpl48l72XeRY+PZQ3iQ4NwZU3Q8e+ittAEEYsy6Pf+LivQswZ6Qc8L145oZiV5vfDMKl";
        sout << "yTIMBjVFFx7o7c+BcJ5sFf4brxrSreMuecv+WUdGJOzrh5MwzmNSlI/baUgzs8viwaKen0zp2GH/";
        sout << "pzdgEN6cfUh+ERADYBRg7oR+OYFBVVexb0SVLsnAnJ6hnbEJ0lBXCSirw8GabPUrJlN+TEtHgdrw";
        sout << "nISwznkUCXI+LuGTuT4N1484bXbi1Y0UUKOKuCHK5/VeJbO95P83+mKEGjyu4Pt84Nr+QCCraZ5y";
        sout << "bDt+bT0En0LsOuMeULXhjPr5f3/EMg2htlpT6yqBgFJPy1MmOqRVNgqZoPW5RA2+xif5c8/hB05f";
        sout << "Sx3BALho7L0olnQAhGZ0Le0WrOqECA8n5DNTkiICVwB37FdhRu7WyRrVL5ezYTKL5gh7AxpGFC/M";
        sout << "ljOzRHFeRBfOFiFbzQEOgyPIqoXeVF/tTIptjia61N7C1lEICIWkNrjlnjvhcE3d1SShCfsWeDk5";
        sout << "QNJrMigws7NYpzykRqsMFDmK7TKsmND8sxc6C8keI5d0hzcPpw5nsDughxWXN0J1qlEX80sZqQbd";
        sout << "FhN8WPRhENjFD6pQP//10Aroh6p0//FZ8UPraJo3u+E9hYCqShqpvm3GsnzGxXJtPTGDuAiIkspi";
        sout << "qOtXirex5WuL8sxyNlHi7COYTNFtHk17m8zZ7qqGC8nr4R0Ds8dOQuHliG+y0Ge/SsH4TAWA76sE";
        sout << "3InZjAwWyfrFz69dFW+CDW08QGD5qV4rFvTBUsuGsO8nNJb+3rjkz7YuR8fpxdBaXW9VcGC2HsE9";
        sout << "afDGJs69Bbsf2G+Zv7wNp+ngcigHT1774BN5fHTzxKnkYDtLk8BEKAhzd0SZYygJbYzgXZD49Bqu";
        sout << "7vK8OMlhssaXxi7B0Xjy7LtROmxz1vAJfmfRi8MXE1qTVJ9tV6rd3fABGx3DgVIz8I0CRi7osWvy";
        sout << "RDdc6A3NKeWH3eEu12BFlF74HCA8683h33ekGW5//ll/MicdT5dgJJkSeyUYNhov7ECrh25hG0wf";
        sout << "LoFnkfPnzY4wYHJTgeUxT2TWuq2frdJhqlw5W8k0gDXr1++K14rQBuIBkP61m3md67+KByz2Lao8";
        sout << "cWdSSAZIxOUJ8KPFADaw7VrDSj3fdRkI9u7qkW4DGtK44bzL0ozy01BmpcOYP4mwoEjxU0i/nYHK";
        sout << "ganBzi6I6MA+LMdHNViF/uEbW99rErftedzUbvmWMp4Ntcxszhhjamx6lxN3Qb5N1vEvAEmhYIWb";
        sout << "XQAnMfg2eBPQwrdVWWBkwtdpjmXqURBhYT1y8bqd/6lMmtiBj/Ehl7cqk0pXEvWQr4/xvnMhHbCh";
        sout << "qZVuAowmTpJwC2I3bX0S9d74QLr9jgMMKgcRrHPgvo+OozC23uHooY7DP08xy1e0SA+0mnn/DvH6";
        sout << "1Rexp+cJWEEWRdpDiXAe9hyRR0soIUrN53BO4G4AdupxakdiBNcmnWpkPhdtykf2KIOaxitg44ko";
        sout << "4pC7OxZPJFfZ96gTn0/bZm3KRRvyXzLKh3+YnkX/Pk2sUzjCy9/53tnZKhTpp+0paf7umF95S3If";
        sout << "CEDlcLc2EXThmHfAuQjkJ2Ud5mYVQAmXUVXe75/XSjUmE6U0CvRKTlbLKHM7Q4tME6G1WLYKrrXq";
        sout << "te3qRAOavjpe+DP2DGQje+ZkYkXlFhvQ9VGnQ/6qPxaTQJYrXhBBaXnHWyMix8Rgea9L8FaLQk/E";
        sout << "Iyxr19h1OUAd4Z9Sw6qT364TQa7273nRj4mI+UKT7XQF1AN3ROnrUn/gY33ZE+toro9gQNaxWch9";
        sout << "bkuuLiyoBtivyA50Ixv4freIx6UAFYWe9AFT3WlXrm/hE5zUMNJuZhRKrax2I3cCX4mX/i4y8jHy";
        sout << "7M2eLeyY9odYodQrtL2bj3eLovmAvm9s32HfVO+okV5HkzkPhczI3gVaPf8uLT4Wx0EDW48OyB2q";
        sout << "3PoEqtvJDg/YK5/ielzlWCUSqG0OkV15UUBhv/Wp2YsenT64v5bjVjHYFPEWq5b/YuCpp2nahHbK";
        sout << "UEBgAggpEk1jIqm8aS1gWUCHb6CkVAB5eAYpTs87DUgrdKzYuuEP5Thvm/oQ9dDzJ6nKEuXeRwG0";
        sout << "P9tTDsUfWGYVhGd8MwpJAy9HqhIB/05e4zekmno9et/i5onZysDTUDQY4kqpz3CGVvKEOHhW+ZwI";
        sout << "cBxt46Sy6DQHFJ77u3hCGX0D7GaVtr9QqJ820UVr5RQHwl13NkF79ZQWwMQTk+caQgX1zTaJ1O95";
        sout << "yeDDo7WydSb4QFHne1+X6cfCOTqOp4pdpwfG36eUiyUGRwRBrRzpiIQLfEfL6sY7cU57e4tnRw7e";
        sout << "+PRYXEoyWxNn/kx11gzQenOyYL20fuwOgiqVjVISUPxoqWTMTeeOFEn/C9syjscp9/e+JqaAJLuO";
        sout << "natp1q0p6tepS94DBSoR4OmttbRLJCkkQiCyO8q+tGCcLhxTeUE/gsCQJx2evNKrd+vL3J2ZS/7Q";
        sout << "5oxvjd3AgKZD0VsXlvZejM55ZYO+drhNo9K9gKP/ZEmzcPNJwgps7WleAft5ZDU92SZ1TzNZUDI4";
        sout << "no/y4P8fOY0GL2djnfYJKYrQwvwMmHDa+9Z9MENCwedbYaslt+mqjTwJnZ8YzEAKqhw/2JZkSjts";
        sout << "wWHPZaEbihIr5DIFlnDr/O0quF4lzVouORvY9zxjRdkkEpmhDO1OaD/s3ohiDJqeivJZNXCtuRxP";
        sout << "iVIL2DHrAKJ/4L5LAN28+6dZ3V7PfuaSjLEUPxatwMkfDHUYLilRIYAjcshJzVdCQfJ+PpLOkh9O";
        sout << "OfO3eQGGT+vfcevowyaCRQd4RErjy1DvIX99SoyRS7GhttATMzpvAc4bxT5OvQ14uJI/h1gJiAN5";
        sout << "5TDB49+/PKYFGaB1kfOjISEujZbditjGcy/YbydM0R8sIYQdpCO0f5zUn8Y1xd31zz954xKWbz58";
        sout << "F1w3sQZ1Kc8Gfs23JwsOWgpypbp+X2Y9sWFUH3JfJW7DZTpo1PGdYFtoDL5eAjv5oQUKloxUsl4C";
        sout << "TGfK6nTvXxb01E3cek6FuFxNURAnfm0aAhtv230tmZOhSs4LjeeErnElP03ieb+fxD/BS/qRG31X";
        sout << "pophLzTgNzn3wilCUZImE2jiLy2pjbS6dQrytvexlEQ2URFS9iXqZExJLx7iHHpHDv5uj4JiAnKq";
        sout << "ZAEUqrVSIuvf8STCk/TycE2+GxSL4Uhakt0nreMutYN0dhgZmCjiI40SSoZCoypcshOo8fsEOLhx";
        sout << "/CHQqra269N8tAyFAWd3/wFFtViRdzm4XlBdSiioaZ+TzYcefyui7i6N7Uiwm+0/CQhrvgSbLyYh";
        sout << "7D+l6y6LCbeUpWmE458CWapKWTzbzPG3GQx3tSAzedSBhEQf/vHPVNKae48NUv/DVGciNt8ZdhMw";
        sout << "67V2lcAZkO8a6vNBMUms7XEVrt+1aBWTqN928QN36D/KAbRWpf/2q3MEnpeh6q8Igf7VvnBH3hza";
        sout << "NImMnA8ARDhVmvGjNTVAAwhS+bZ6wU2L+X6XB32exA68cBEI9ttMoO0W9ohTOHo9w38KAWpbc94q";
        sout << "Lrxo1hckWewBBqfKS1CXUnObrQmcCMeScHIhoESomkAh3x+cVgVnw8evDZzP70V1wlT7KbqSxmp+";
        sout << "jpR1wgK7KaxyTYx8Kisp7r0M2Z6J3hxYCNlF3/NQOUJN/PJLc4ryVCGy7gJPCU8P5PXrmzwSuUu2";
        sout << "WF0TANCwhnP5KGiDx1AdZK3/MKCWmegtfypubhEdSOSqwS2dlTTndFD0Q0irf90ZlEE3QEIZIX2P";
        sout << "ZI6IP9y+RgXCuWhtclBJx504x8fb0r58FsEh8aPzC4D6gD3qeMwOE4d/OX5iTrQnqCpgK4XsV2KD";
        sout << "/WFNlZ1eplQbXYiiKpwT2zZsZzjAlfZ1H58J1ct2Uf3vfYwlCiObj7aT4NRKYnM7VKqLE9LKOYab";
        sout << "IIRDxRcBP2PmqYaYodH3ThVjY+Nn9uL97HLq4LnHiWc8rXggmK03/TKliUawlS2C3qPuZKTi+5Ad";
        sout << "Jk4GLFiVchXChvJ0Kehm71f7OKMIi/mZ6utiKNnUpGDz+NKvG3yp+kZx6nwQixHbbJlbko6crx7w";
        sout << "t8Zp4mRfcR3l5TJymW7FiqXOdQN3YEL3jjmdqQ7e41mUdMGjgeklaZQH10dGZ7bjVyYIKScOetHI";
        sout << "TZYBm5PdluilGSbJ0Uhhnt1fRJtfFY0u86MmG7y7VBvWZ7BXKyUGrPlznXKwlqG9aZ1raT2Mmomv";
        sout << "oxhi+U0C9fej+wHfGZVDyh+i7nyX+tNNe4KvZlvc1Xm28wSx4QwZ73KW+ynrYpqGCK4nyEQ6o3+p";
        sout << "fRSrJ9+/++xzxlkc7MsWwtSBmkE9AsB54b079nWtKea0ECue20HGsfukdUZmmqO0ZAsAjxhtC0/F";
        sout << "LGNCaGA+Zv/0YEF73Hp5ByQUEqXdJMH29/uUuNkNrVw8iq1m+FARL1L9Ajc6dMMvbZXVVjTzUlme";
        sout << "EZ34/9a5kLwRLN3qBUt6bwDSZjm103B9Ygw2yxXJb2FnfdkhQvoR7lx7w9xmmr7YYt0L+UHI/bLo";
        sout << "yTsdg+2AS3KmzkICSTfP99e+U6uai9Is7xYWNPQ1FJ7qkrWdRpW3WKIkBE4RwC0qCib034/dQjrY";
        sout << "BO+fWbspfdqU0mchgYyxnaDU0bN0fwHd6WOooaqkllSdf6W9e5aUYWDse8Vt9lo+iQoTyytoWVo7";
        sout << "63sxCymj6P/dc3PQ84DDagvr4SfKLbYGobLxnXoopbXlhJ672kbs0rITseKx5xoVEIDqo9fprmM6";
        sout << "7KB7PAPet/4SrFpKscliUXWIxRiw7KPHYzwqg8i+iZagmVMocotnFtPTCD1xYHCPZjrhlF1CxZmP";
        sout << "0/pXP+15KMKHJluPduF6W/LkUNrQ1IJN2hqTsl5FwDv9bzhp4LtkNbsbZv2gA7/vJIHiB3btX2tg";
        sout << "27g3rvuhGNk3U5MuWLWlT2HY0oiRkFhxBYWjP1MI1/Hc+MPT3QAdAlPp9mSPgByFY3yJ4XWqy5Lw";
        sout << "eSsDUxYk26cKrFO3QPAam1JMStYUrEaXGckFLV7kqNsfUfCbIJno45XNMylNOuAtgVHNAbQYzQTn";
        sout << "5LQMpl97ll+5OfqqOR2aWEP+RUue55dgERWRBZH0yLtP69GTYFffpOVZnEXoyv5pJI/SPASxofWt";
        sout << "fYVHdhdbfvy2xkQvg/5wgOXEBZWFujkxDmbtPG3FU3/UTA406GpZsRilwDsEcyxkQ3IX7lM9iVx7";
        sout << "WlmugbFAhi52IILozdIliZGWN1leeaVBELkwznpo+zDcEamUPJMpCAaj/RtBXYz63rqjJZ8xhHlc";
        sout << "mUP9ECb4acsE1i7e0vX0Qej8wgyGsaSfVPdAHV74FT1agtIagzpL+hDrALQ1sX+bheYaiCxhztqa";
        sout << "HrBQG/vtodb7ABaybo0pHX0YQ/8XCi+yxaYewl1iagAPOzYOYY1eEpw1IG78/doxr89fHpi2bvrR";
        sout << "WmIlgjUOT3BX1OcLXCq3fVNiQ/JTBX/kgPcLhD4W+GywcV0P5E8n82xVel0P/nKwca2YdHa9A2XO";
        sout << "yHXaMrXax1sn5PqobEqf96DFS8LkiueThxpFtcV12RscyL+tNzEL6SP2mTmnn36SfxhKRFar4LPw";
        sout << "b5NyutyXoUlvZ0xbQQ6r94TlSThjV8/BFqjqd/UQD56+GZVWH1bBBrd6DA5ktchYYY1UwrLZ79ud";
        sout << "1z10elvHy/NUAkJYvsGFIydc7b7J/5b3ev2CRNcqBXWlxfcAr9dSeB6MgLHSmhLKln5NSOChzKl7";
        sout << "cxI8SVst0OMfaLqsG80Pb8vFqbugTh98HafuiG4/8jY7ZK1ChBS054PqImLQ89zEp29TJmGVsC0Z";
        sout << "0lLRcikxw+8kpGP6SKIf2uAbMR77RX9//yYkpmLWetHnQ/fAHou637J3kWMwuxC2PtCpnurhG4d2";
        sout << "ACDIN8PaWOmPcdWXHgAj2mgQdlXu5Pbo/RA6Evogx3udHxBkAECvWAx3nkQCqPBZc6Rlw1DHC3NZ";
        sout << "//zy4zNYqlw9s+z5byg0rCkasnI6K7AvEW9xUh5QMeGN1Of11FTa5rMke+uheLxXZuCIRzsZp8ee";
        sout << "ZaQ5WGyj5/d7k7rcdDdXlA2JFuP36I07RPF5x0NLnpcQBFNc+5jWm5XXOUCjQgs8DQHA+L1aJKVV";
        sout << "JTVnTet2oydKG+wc4601eul2oNLRlSVeUJDBjydf48NLAvzDRX9tBoSFU0cwG6rgX4E0uM0l9G/7";
        sout << "OtzqUFW0E2Uf8YYkrm+a3AmC3QPJsUYI2ZE/ttID3bj8gHiJ8oxFsbUDpzR8MoqJcKeFH7UktiZ8";
        sout << "AUyx8i52UtRLM3zItoQky7xhs+w84qGr5xC5iOjyvUbWEtz/Ri1o8wysbTnBkmaIIGdHhsXlTp6h";
        sout << "DxJG2ZUqNKi0xESkD4nIrllgg2MsTAPlkDNtunlI7aQrHokL1uUrHgjb3lSdIIKKGhjw/xRFXahE";
        sout << "bMKOFXueOXldqwS98TxHSULmf0lhyJJ2X0Mx1WLwCG/erjqP07kK2/SCZy5CsObeYd9OnlIZ4Pxx";
        sout << "v18VlK/vSsLqP5MB0hzx8mMarxegk1PMwUOQknO+1g2IQ8vtmPCCbXuNc4+6aQCXSvS0i2s7cBIg";
        sout << "LHm8bCO7DRx3RcBpSgH2cdd7UZY/8F0bgkEGYtlN9DsOO/JD7dmz11dH/s3ElX95RnhlOKBIbkSl";
        sout << "Gi3q8NheetdpHQFe0QKb5q8oCLtWZ2VEzxo3SEF/HMASirSI6HA/mQ5t0zl8PPYipFOBauGMkFWO";
        sout << "QFc14V1t+ukyeuVnbf9t0weBPxE7xM6rqpn0EPhqk1U40iwy8n8ed8ci0VDV4CPr+SJWku+2voji";
        sout << "jD+CVSdoCwIsWDesba7O+vEp247pGVwERlYSArWLiOxRYPhpVKpiOsQGa4UMn1Yh+3+jUOte40OM";
        sout << "/CBW6WKZNfPSkBb153Tint3IWhcs83OIO4wW7WBJCYpLj1ub7NxnsTtSAodzfdot7dKdDgcT9JO+";
        sout << "62WjxBRYZQBqA1G351CaKPJSeXGExYsagOaNmWCi/yNaNoGigYn+JMifBZjxCL6zs6av4p9XkjiG";
        sout << "JBjqYtogaDjTH6ZDsqw87I5ytpkFq1rk2Q3ml6pMzPYM+3CacoT6lCrkI5/KwApH3QfNlc7pXxUN";
        sout << "Op+AFt5GnaJvEFCd4m3s3qvRtKvhL+RBB+3Wc7bKBkTfdGYaIjT1x5Xtii6yRNrFIHChVjNI0n5f";
        sout << "D1fp8+2KsdCvXVL4QmJRi8EPyKkGcB9SzENIS0ljzK+VaLzJGZxrUVQEW95hJ/4wPE1TYC4/+rG2";
        sout << "QCuWmjES+8CkHEgalY6uwoXfn+yR/gpgAxS5flSJaWBiTA0cAA==";
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

    case dataset_id::BLACK_HOLE_ARTICLE:
        return detail::decompress_data(datasets::get_blackhole_article_compressed());

    case dataset_id::PHYSICS_PARAGRAPHS:
        return detail::decompress_data(datasets::get_physics_paragraphs_compressed());

    case dataset_id::BLACK_HOLE_QA_PARTA:
        return detail::decompress_data(datasets::get_blackhole_qa_pa_compressed());

    case dataset_id::BLACK_HOLE_QA_PARTB:
        return detail::decompress_data(datasets::get_blackhole_qa_pb_compressed());

    case dataset_id::BLACK_HOLE_QA_PARTC:
        return detail::decompress_data(datasets::get_blackhole_qa_pc_compressed());

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
        auto qa_pairs = get_dataset_as_pairs(dataset_id::BLACK_HOLE_QA_PARTA);
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
