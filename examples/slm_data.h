#ifndef SLM_DATA_H
#define SLM_DATA_H

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

#include <dlib/compress_stream.h>
#include <dlib/base64.h>

// Text parts for training (from the "Tiny Shakespeare" text input)
// Used by the <slm_basic_train_ex.cpp> example
const std::string get_shakespeare_extract()
{
    dlib::base64 base64_coder;
    dlib::compress_stream::kernel_1ea compressor;
    std::ostringstream sout;
    std::istringstream sin;

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

    sin.str(sout.str());
    sout.str("");

    base64_coder.decode(sin, sout);
    sin.clear();
    sin.str(sout.str());
    sout.str("");

    compressor.decompress(sin, sout);
    return sout.str();
}

// Full text for training
const std::string shakespeare_text = get_shakespeare_extract();

// Prompt for text generation
const std::string shakespeare_prompt = R"(QUEEN ELIZABETH:
But thou didst kill my children.

KING RICHARD III:
But in your daughter's womb I bury them:
Where in that nest of spicery they shall breed
Selves of themselves, to your recomforture.

QUEEN ELIZABETH:
Shall I go win my daughter to thy will?

KING RICHARD III:
And be a happy mother by the deed.

QUEEN ELIZABETH:
I go. Write to me very shortly.
And you shall understand from me her mind.

)";

// Retrieves the content of an internal "plain text" dataset
// Used by the <slm_advanced_train_ex.cpp> example
const std::string get_internal_dataset()
{
    dlib::base64 base64_coder;
    dlib::compress_stream::kernel_1ea compressor;
    std::ostringstream sout;
    std::istringstream sin;

    // Black holes: physics and observations
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

    sin.str(sout.str());
    sout.str("");

    base64_coder.decode(sin, sout);
    sin.clear();
    sin.str(sout.str());
    sout.str("");

    compressor.decompress(sin, sout);
    return sout.str();
}

// Question-answer pairs dataset on black holes physics and observations
// Each pair contains a question (potentially with context) and its corresponding answer
std::vector<std::pair<std::string, std::string>> qa_pairs = {
    {"What is a black hole?", "A black hole is a region of spacetime where gravitational forces are so intense that nothing, not even light, can escape once it crosses the event horizon. It represents one of the most extreme objects predicted by Einstein's general theory of relativity."},
    {"Can light escape from a black hole?", "No, once light crosses the event horizon, it cannot escape."},
    {"Who coined the term black hole?", "Physicist John Wheeler coined the term in 1967, though the concept had been explored much earlier."},
    {"When was Einstein's general theory of relativity published?", "Einstein published his general theory of relativity in 1915."},
    {"What is the event horizon?", "The event horizon is the boundary around a black hole beyond which escape becomes impossible. It marks the point of no return for anything approaching the black hole."},
    {"How are stellar-mass black holes formed?", "Stellar-mass black holes form from the gravitational collapse of massive stars at the end of their life cycles. When a star with mass greater than approximately 20-25 solar masses exhausts its nuclear fuel, the core collapses while the outer layers explode in a supernova."},
    {"What is Sagittarius A*?", "Sagittarius A* is the supermassive black hole at the center of our Milky Way galaxy, with a mass of about 4 million solar masses."},
    {"Are supermassive black holes common?", "Yes, supermassive black holes reside at the centers of most galaxies, with masses ranging from millions to billions of solar masses."},
    {"What are the three parameters that completely characterize a black hole?", "Mass, electric charge, and angular momentum are the three parameters that completely characterize a black hole according to the no-hair theorem."},
    {"Scientists have observed that galaxies typically have supermassive black holes at their centers. What might explain how these enormous objects formed in the early universe?", "Several theories exist including direct collapse of massive gas clouds in the early universe, hierarchical mergers of smaller black holes, or runaway stellar collisions in dense stellar clusters. The exact formation mechanism remains an active area of research."},
    {"What is the Schwarzschild radius?", "The Schwarzschild radius defines the event horizon for a non-rotating black hole, located at approximately 3 kilometers times the mass in solar masses."},
    {"Do all black holes rotate?", "Most black holes are expected to rotate to some degree, described by the Kerr metric. However, the theoretical non-rotating Schwarzschild black hole provides a useful simplified model."},
    {"What happens at the singularity?", "The singularity represents a point of infinite density at the center of a black hole where the laws of physics as we currently understand them break down."},
    {"What is spaghettification?", "Spaghettification is the process where extreme tidal forces near a black hole stretch and compress objects, potentially tearing them apart as they approach the event horizon."},
    {"Would you be torn apart immediately if you approached any black hole?", "Not necessarily. For supermassive black holes with very large event horizons, tidal forces at the horizon can be surprisingly gentle, allowing objects to cross intact. However, stellar-mass black holes would tear objects apart long before reaching the event horizon."},
    {"What is Hawking radiation?", "Hawking radiation is thermal radiation predicted by Stephen Hawking in 1974 that black holes emit due to quantum effects near the event horizon, arising from particle-antiparticle pair creation."},
    {"If black holes are black, how did astronomers obtain an image of one?", "The Event Horizon Telescope captured the shadow of the black hole surrounded by bright emission from the accretion disk, the superheated material orbiting and falling into the black hole. The image shows the dark central region against the bright ring of emission."},
    {"When was the first direct image of a black hole obtained?", "The first direct image of a black hole was produced by the Event Horizon Telescope collaboration in April 2019, showing the supermassive black hole at the center of galaxy M87."},
    {"What is an accretion disk?", "An accretion disk is a structure formed when matter spirals into a black hole, heating up through friction and gravitational compression to temperatures of millions of degrees and emitting intense electromagnetic radiation."},
    {"Do black holes eventually evaporate?", "Yes, Hawking radiation leads to black hole evaporation over incredibly long timescales. A solar-mass black hole would require approximately 10 to the 67 years to evaporate completely, far exceeding the current age of the universe."},
    {"What was GW150914?", "GW150914 was the first direct detection of gravitational waves by LIGO in September 2015, involving two merging black holes with masses of approximately 36 and 29 solar masses."},
    {"How much energy was released during the GW150914 merger?", "Three solar masses worth of energy was radiated as gravitational waves in a fraction of a second during the merger."},
    {"Can we see black holes directly with telescopes?", "No, black holes emit no light themselves. We detect them through their gravitational effects on nearby matter and through the electromagnetic radiation emitted by material falling into them."},
    {"What are relativistic jets?", "Relativistic jets are narrow beams of matter and energy ejected from the poles of some accreting black holes at velocities approaching the speed of light, extending for thousands or millions of light-years."},
    {"What is an active galactic nucleus?", "An active galactic nucleus is a region at the center of a galaxy powered by a supermassive black hole accreting material at enormous rates, often outshining the entire host galaxy."},
    {"Who derived the first exact solution to Einstein's field equations for black holes?", "Karl Schwarzschild derived the first exact solution in 1916, describing the geometry of spacetime around a non-rotating spherical mass."},
    {"What is the information paradox?", "The information paradox is the apparent conflict between Hawking radiation and quantum mechanics' requirement that information cannot be destroyed, representing one of the deepest unsolved problems in theoretical physics."},
    {"What is the ergosphere?", "The ergosphere is a region outside the event horizon of a rotating black hole where spacetime itself is dragged around the black hole, unique to rotating Kerr black holes."},
    {"Can anything escape from inside the event horizon?", "No, nothing can escape from inside the event horizon, not even light or information."},
    {"Why are primordial black holes candidates for dark matter?", "Primordial black holes could have formed in the extremely high-density conditions immediately after the Big Bang and might account for some or all of the dark matter in the universe, though they remain undetected."},
    {"Imagine you're an astronaut approaching a black hole while your colleague watches from a safe distance far away. What would each of you observe about the passage of time?", "You would experience time normally, but your colleague would see your clock slowing down dramatically as you approach the event horizon, effectively appearing to freeze at the horizon. This extreme time dilation is a consequence of the intense gravitational field warping spacetime."},
    {"What is the no-hair theorem?", "The no-hair theorem states that black holes have no distinguishing features beyond mass, electric charge, and angular momentum, meaning all other information about the matter that formed them is lost."},
    {"Who predicted Hawking radiation?", "Stephen Hawking predicted Hawking radiation in 1974."},
    {"What is the temperature of Hawking radiation from a solar-mass black hole?", "The temperature is extremely low, far below the cosmic microwave background, making it essentially undetectable for stellar-mass and supermassive black holes."},
    {"What was the mass of the supermassive black hole imaged in M87?", "The M87 black hole has a mass of approximately 6.5 billion solar masses."},
    {"In the early universe, astronomers have discovered quasars powered by billion-solar-mass black holes existing less than a billion years after the Big Bang. Does this create a problem for our understanding of black hole formation?", "Yes, this is a significant puzzle. Growing black holes to such enormous masses so quickly challenges our theoretical models. Researchers are exploring various scenarios including more efficient accretion mechanisms or different initial seed mass configurations. Would you like to know more about the proposed solutions?"},
    {"What distinguishes a Kerr black hole from a Schwarzschild black hole?", "A Kerr black hole rotates while a Schwarzschild black hole does not. Kerr black holes have additional structure including an ergosphere and potentially a ring singularity rather than a point singularity."},
    {"Can black holes merge with each other?", "Yes, black holes can merge through the emission of gravitational waves, producing the most powerful gravitational wave signals detectable by current instruments during their final stages."},
    {"What role do black holes play in galaxy evolution?", "Black holes regulate galaxy evolution through feedback from active galactic nuclei, which can heat or expel gas, preventing it from cooling and forming new stars. The strong correlation between supermassive black hole masses and galaxy bulge properties suggests a deep connection."},
    {"Are intermediate-mass black holes real?", "Recent observations suggest their existence in globular clusters and dwarf galaxies, though they have been more elusive than stellar-mass or supermassive black holes."},
    {"How does general relativity explain black holes?", "General relativity describes gravity as the curvature of spacetime caused by mass and energy. Black holes represent extreme curvatures where this warping becomes so severe that it creates a region from which nothing can escape."},
    {"When did John Michell and Pierre-Simon Laplace propose the idea of dark stars?", "They proposed the concept in the 18th century using Newtonian mechanics, suggesting objects so dense that light could not escape, predating the term black hole by nearly two centuries."},
    {"What happens to time at the event horizon from an external observer's perspective?", "Time appears to stop at the event horizon from an external observer's perspective, with objects appearing to freeze as they approach it."},
    {"How large is the event horizon of a one-solar-mass black hole?", "The event horizon would be located at approximately 3 kilometers from the center."},
    {"Given that we cannot see black holes directly and they represent exotic physics, how confident are scientists that they actually exist?", "Scientists are highly confident. Multiple independent lines of evidence including gravitational wave detections, direct imaging of black hole shadows, observations of stellar orbits around Sagittarius A*, and X-ray emissions from accretion disks all consistently point to the existence of black holes as described by general relativity."},
    {"What determines whether a collapsing star will form a black hole or a neutron star?", "The mass of the collapsing core determines the outcome. Cores below approximately 2-3 solar masses form neutron stars, while more massive cores collapse to form black holes."},
    {"Do black holes grow over time?", "Yes, black holes can grow by accreting matter from their surroundings or by merging with other black holes, though they can also slowly evaporate through Hawking radiation."},
    {"What made the 2015 LIGO detection so revolutionary?", "It was the first direct detection of gravitational waves and the first direct confirmation of black hole mergers, opening an entirely new way to observe the universe and providing strong-field tests of general relativity."},
    {"How do we know there is a supermassive black hole at the center of our galaxy?", "Astronomers have tracked the orbits of stars near the galactic center for decades. These stars move at incredible speeds in elliptical orbits, indicating they are orbiting an invisible object with a mass of about 4 million solar masses concentrated in a very small region."},
    {"What are X-ray binaries?", "X-ray binaries are systems where a black hole or neutron star accretes matter from a companion star, with the infalling matter heating up and emitting intense X-rays that reveal the presence of the compact object."},
    {"Is it possible for a black hole to have zero rotation?", "Theoretically yes, described by the Schwarzschild solution, but in practice, black holes are expected to have at least some angular momentum inherited from the material that formed them or was accreted."},
    {"What happens to the matter that falls into a black hole?", "Once matter crosses the event horizon, it continues falling toward the singularity where it contributes to the black hole's mass. From outside, we can only observe the matter's contribution through changes in the black hole's mass, charge, and angular momentum parameters."},
    {"Scientists often say that inside a black hole, space and time switch roles. What does this mean?", "This is a consequence of the extreme spacetime curvature. Inside the event horizon, the radial direction toward the singularity becomes timelike, meaning moving toward the singularity becomes as inevitable as moving forward in time outside the black hole. However, the full implications remain difficult to visualize and interpret."},
    {"What is frame dragging?", "Frame dragging is the effect where a rotating mass drags spacetime around with it. Near rotating black holes, this effect is so strong that it creates the ergosphere where even light must orbit the black hole."},
    {"Can we extract energy from a black hole?", "Yes, through mechanisms like the Penrose process in the ergosphere of a rotating black hole, or by carefully controlling accretion. In principle, up to 29% of the rest mass energy can be extracted from a maximally rotating black hole."},
    {"Why don't we observe Hawking radiation from astrophysical black holes?", "The temperature of Hawking radiation is inversely proportional to mass, making it vanishingly small for astrophysical black holes. It would be completely overwhelmed by the cosmic microwave background and other sources of radiation."},
    {"What is a maximally rotating black hole?", "A maximally rotating or extremal Kerr black hole has the maximum possible angular momentum for its mass, beyond which the event horizon would disappear, violating the cosmic censorship hypothesis."},
    {"How long would it take to fall from the event horizon to the singularity of a stellar-mass black hole?", "For a stellar-mass black hole, the free-fall time from the event horizon to the singularity would be only a fraction of a millisecond as experienced by an infalling observer."},
    {"What is the cosmic censorship hypothesis?", "The cosmic censorship hypothesis, proposed by Roger Penrose, suggests that singularities are always hidden behind event horizons and cannot be observed from the outside universe, though this remains unproven."},
    {"If I orbited very close to a black hole and then returned to Earth, would I have aged differently than people who stayed behind?", "Yes, due to gravitational time dilation, you would have aged less than people on Earth. The closer you orbit to the black hole and the longer you stay, the more pronounced this effect becomes."},
    {"What is the innermost stable circular orbit?", "The innermost stable circular orbit is the smallest radius at which a particle can stably orbit a black hole. For a non-rotating black hole, this is at three times the Schwarzschild radius, while for rotating black holes it can be closer."},
    {"Can black holes destroy information?", "This is the heart of the information paradox. Quantum mechanics says no, but Hawking radiation appears to suggest yes. Recent theoretical work suggests information may be preserved in subtle correlations in the Hawking radiation, but the issue is not fully resolved."},
    {"What happens if two supermassive black holes merge?", "They create an even more massive black hole and emit tremendous amounts of energy in gravitational waves. Such mergers are expected in galaxy collisions and may be detectable by future space-based gravitational wave observatories like LISA."},
    {"What is gravitational lensing by black holes?", "Black holes bend light passing nearby due to their intense gravity, acting as lenses that can magnify, distort, or create multiple images of background objects. This effect was crucial in confirming general relativity."},
    {"Are there any limits to how massive a black hole can be?", "Theoretically, there appears to be no upper limit, though the most massive observed black holes are around 10-20 billion solar masses. Practical limits may arise from available material to accrete and galaxy formation processes."},
    {"What are quasi-periodic oscillations?", "Quasi-periodic oscillations are nearly periodic variations in X-ray brightness from accreting black holes and neutron stars, thought to arise from instabilities in the accretion flow or from material orbiting very close to the black hole."},
    {"Could the universe itself be inside a black hole?", "Some speculative theories have explored this idea, but there is no strong evidence for it. The large-scale properties of our universe appear quite different from the interior of a black hole as described by general relativity."},
    {"What is the photon sphere?", "The photon sphere is a region outside the event horizon where gravity is strong enough that photons can orbit the black hole in circular paths. For a non-rotating black hole, it is located at 1.5 times the Schwarzschild radius."},
    {"How do binary black hole systems form?", "Binary black hole systems can form through several channels including the evolution of massive binary star systems where both stars collapse to black holes, or through dynamical capture in dense stellar environments like globular clusters."},
    {"I have read that black holes can spin at nearly the speed of light. How is this possible for such massive objects, and what prevents them from spinning faster?", "The event horizon of a maximally rotating black hole moves at the speed of light. They cannot spin faster because that would require the event horizon surface to move faster than light, which is impossible. The rotation is characterized by angular momentum per unit mass rather than a physical spinning surface. More precise answers would require specifying the exact quantity you are interested in."},
    {"What is the firewall paradox?", "The firewall paradox, proposed in 2012, suggests that to preserve quantum mechanics, the event horizon might be a wall of high-energy particles that would destroy anything crossing it, contradicting general relativity's prediction of an uneventful horizon crossing for large black holes."},
    {"Can we ever reach a black hole singularity and survive?", "No, any object approaching the singularity would be torn apart by infinite tidal forces and crushed to infinite density. Moreover, the singularity represents a breakdown of our physical theories, so we cannot reliably predict what would happen there."},
    {"What is an extremal black hole?", "An extremal black hole is one at the theoretical limit of its parameters, such as maximum rotation for a given mass or maximum electric charge. These represent boundary cases in black hole physics with interesting theoretical properties."},
    {"How precise were the tests of general relativity provided by gravitational wave observations?", "The LIGO observations have tested general relativity with unprecedented precision in the strong-field regime, confirming predictions to within the measurement uncertainties and ruling out many alternative theories of gravity."},
    {"What distinguishes a black hole from other compact objects like neutron stars?", "The key distinction is the presence of an event horizon. Black holes have event horizons from which nothing can escape, while neutron stars have solid surfaces that can be observed and do not trap light."},
    {"Researchers sometimes discuss white holes as the time reverse of black holes. Do white holes exist in nature?", "White holes are theoretical objects predicted by general relativity as time-reversed black holes that expel matter rather than consuming it. However, there is no observational evidence for white holes, and they would be highly unstable if they existed."},
    {"What is the Eddington limit?", "The Eddington limit is the maximum luminosity an accreting object can achieve when radiation pressure from emitted light balances the inward gravitational force. Black holes accreting above this limit can blow away infalling material."},
    {"Can black holes collide at speeds approaching the speed of light?", "Yes, in the final moments before merger, the black holes in a binary system orbit each other at velocities that can be a significant fraction of the speed of light, with their event horizons eventually merging."},
    {"What is a naked singularity?", "A naked singularity would be a singularity not hidden behind an event horizon, visible to external observers. The cosmic censorship hypothesis suggests these cannot form in realistic collapse scenarios, but this remains unproven."},
    {"How do we distinguish between different types of black holes observationally?", "We primarily distinguish them by mass: stellar-mass black holes are a few to tens of solar masses, intermediate-mass are hundreds to thousands, and supermassive are millions to billions. We also observe different formation mechanisms and environments for each type."},
    {"What happens to electric charge on a black hole?", "According to the no-hair theorem, a black hole can carry electric charge as one of its three defining properties, but astrophysical black holes are expected to be nearly electrically neutral because they would quickly attract opposite charges from their surroundings."},
    {"Some physicists have proposed that black holes could be portals to other universes or regions of spacetime. Is there any evidence for this?", "These ideas arise from mathematical extensions of black hole solutions in general relativity, such as wormholes connecting to white holes or other universes. However, these extensions are highly speculative, and there is no observational evidence. Most would be unstable or require exotic matter with negative energy. Would you like to explore the theoretical basis for these ideas?"},
    {"What is the membrane paradigm?", "The membrane paradigm is a useful theoretical framework where the event horizon is treated as a physical membrane with properties like electrical conductivity and viscosity, simplifying calculations of black hole behavior."},
    {"How quickly can matter fall into a black hole?", "The rate of accretion is limited by factors including the Eddington limit and angular momentum. Matter typically spirals in through an accretion disk over timescales ranging from days to years depending on the system."},
    {"What are black hole oscillations?", "When perturbed, black holes oscillate at characteristic frequencies called quasi-normal modes, somewhat like ringing bells. These oscillations depend on the black hole's mass and spin and are imprinted in gravitational wave signals."},
    {"Can we use black holes for time travel?", "Rotating black holes have theoretical solutions called Kerr wormholes that could allow closed timelike curves, but these require passing through the inner horizon where physics becomes uncertain and likely unstable. Practical time travel using black holes remains firmly in the realm of speculation."},
    {"What is the area theorem?", "Hawking's area theorem states that the total surface area of event horizons in an isolated system never decreases, analogous to entropy in thermodynamics. This led to the development of black hole thermodynamics."},
    {"How do supermassive black holes affect their host galaxies?", "They regulate star formation through feedback, energize the interstellar medium through jets and radiation, and their growth history is intimately connected to galaxy evolution. The correlation between black hole mass and galaxy properties suggests coevolution."},
    {"Could a micro black hole be created in particle accelerators?", "Some theories with extra dimensions predict micro black holes could be produced in high-energy collisions at accelerators like the LHC. If created, they would evaporate instantly via Hawking radiation. However, none have been detected, placing constraints on such theories."},
    {"What is the black hole entropy?", "Black hole entropy is proportional to the event horizon's surface area, given by the Bekenstein-Hawking formula. This connects black hole physics to thermodynamics and information theory, though the microscopic origin of this entropy remains an open question."},
    {"Do black holes violate the second law of thermodynamics?", "Initially, it appeared they might by allowing entropy to disappear inside. However, Bekenstein and Hawking showed that black holes have entropy proportional to their horizon area, and including this in calculations preserves the second law."},
    {"What are primordial black holes and how do they differ from stellar black holes?", "Primordial black holes hypothetically formed from density fluctuations in the early universe rather than stellar collapse. They could have a wide range of masses, including very small ones, unlike stellar black holes which have a minimum mass around three solar masses."},
    {"Given the extreme conditions near black holes, could exotic physics beyond general relativity become important?", "Yes, near the singularity and possibly even near the horizon of small black holes, quantum gravitational effects should become important. This is one reason studying black holes is important for developing theories of quantum gravity like string theory or loop quantum gravity."},
    {"What is the shadow of a black hole?", "The black hole shadow is the dark region in images like the M87 observation, representing the region where photons are captured by the black hole or bent so strongly they circle multiple times. It is about 2.6 times larger than the event horizon for a non-rotating black hole."},
    {"Can anything orbit inside the event horizon?", "No, inside the event horizon, all trajectories lead inevitably to the singularity. The concept of orbiting requires stable or periodic motion, which is impossible once inside."},
    {"What is the relationship between black holes and quasars?", "Quasars are extremely luminous active galactic nuclei powered by supermassive black holes accreting matter at very high rates in the early universe. They represent a phase of intense black hole growth and can outshine entire galaxies."},
    {"How do we measure the spin of a black hole?", "Black hole spin can be measured through several methods including analyzing the X-ray spectrum from the inner accretion disk, studying the broadening of iron emission lines, or from gravitational wave signals during mergers."},
    {"Are there any practical applications of black hole physics?", "While we cannot directly use black holes, studying them has advanced our understanding of gravity, spacetime, and quantum mechanics. Technologies like GPS rely on general relativity, and black hole research drives development of detectors and computational methods with broader applications."},
    {"What would I see if I fell into a black hole?", "You would see the external universe increasingly blueshifted and time-dilated as you approach the horizon. After crossing, you would continue falling toward the singularity with increasingly extreme tidal forces. The view would depend on the black hole's size and rotation."},
    {"Could dark matter be composed of primordial black holes?", "This is an active area of research. Primordial black holes are candidates for some or all dark matter, though various observational constraints limit the possible mass ranges. Recent gravitational wave observations have renewed interest in this possibility."},
    {"What is accretion flow?", "Accretion flow refers to the pattern of matter falling onto a compact object like a black hole. It can take various forms including thin disks, thick disks, and hot radiatively inefficient flows, depending on the accretion rate and angular momentum."},
    {"How do magnetic fields behave near black holes?", "Magnetic fields can be amplified and twisted by the rotating spacetime and accretion flow, playing a crucial role in launching jets and regulating accretion. The magnetorotational instability drives turbulence in accretion disks."},
    {"I have heard conflicting accounts about whether black holes destroy information or preserve it. What is the current scientific consensus on this issue?", "This remains one of the most debated topics in theoretical physics. Many physicists now lean toward information being preserved, possibly encoded in Hawking radiation or the horizon structure, but there is no consensus on the precise mechanism. The resolution likely requires a complete theory of quantum gravity."}
};

std::string build_qa_text(const std::vector<std::pair<std::string, std::string>>& qa_pairs)
{
    std::ostringstream oss;
    oss << "\n\n";
    for (const auto& pair : qa_pairs)
        oss << pair.first << " " << pair.second << "\n";
    return oss.str();
}

#endif // SLM_DATA_H
