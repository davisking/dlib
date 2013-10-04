// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/image_transforms.h>
#include <vector>
#include <sstream>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>
#include <dlib/image_io.h>

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    dlib::logger dlog("test.fhog");


    class fhog_tester : public tester
    {
    public:
        fhog_tester (
        ) :
            tester (
                "test_fhog",       // the command line argument name for this test
                "Run tests on the fhog functions.", // the command line argument description
                0                     // the number of command line arguments for this test
            )
        {
        }

        template <typename image_type>
        void test_fhog_interlaced(
            const image_type& img,
            const int sbin,
            const array2d<matrix<float,31,1> >& ref_hog
        )
        {
            array2d<matrix<float,31,1> > hog;
            extract_fhog_features(img, hog, sbin);

            DLIB_TEST(hog.nr() == ref_hog.nr());
            DLIB_TEST(hog.nc() == ref_hog.nc());
            for (long r = 0; r < hog.nr(); ++r)
            {
                for (long c = 0; c < hog.nc(); ++c)
                {
                    DLIB_TEST_MSG(max(abs(hog[r][c] - ref_hog[r][c])) < 1e-6, max(abs(hog[r][c] - ref_hog[r][c])));
                }
            }
        }

        template <typename image_type>
        void test_fhog_planar(
            const image_type& img,
            const int sbin,
            const array2d<matrix<float,31,1> >& ref_hog
        )
        {
            dlib::array<array2d<float> > hog;
            extract_fhog_features(img, hog, sbin);

            DLIB_TEST(hog.size() == 31);
            for (long o = 0; o < (long)hog.size(); ++o)
            {
                DLIB_TEST(hog[o].nr() == ref_hog.nr());
                DLIB_TEST(hog[o].nc() == ref_hog.nc());
                for (long r = 0; r < hog[o].nr(); ++r)
                {
                    for (long c = 0; c < hog[o].nc(); ++c)
                    {
                        DLIB_TEST_MSG(std::abs(hog[o][r][c] - ref_hog[r][c](o)) < 1e-6, std::abs(hog[o][r][c] - ref_hog[r][c](o)));
                    }
                }
            }
        }

        void test_on_small()
        {
            print_spinner();
            array2d<unsigned char> img;
            dlib::array<array2d<float> > hog;

            // do this just to make sure it doesn't crash on small images
            for (int i = 0; i < 10; ++i)
            {
                img.set_size(i,i);
                assign_all_pixels(img, i);
                extract_fhog_features(img, hog);
            }
            for (int i = 1; i < 10; ++i)
            {
                img.set_size(i,i+1);
                assign_all_pixels(img, i);
                extract_fhog_features(img, hog);
            }
            for (int i = 1; i < 10; ++i)
            {
                img.set_size(i+1,i);
                assign_all_pixels(img, i);
                extract_fhog_features(img, hog);
            }
        }

        void perform_test (
        )
        {
            test_on_small();

            print_spinner();
            // load the testing data
            array2d<rgb_pixel> img;
            array2d<unsigned char> gimg;
            istringstream sin(get_decoded_string_face_dng());
            load_dng(img, sin);
            assign_image(gimg, img);
            sin.str(get_decoded_string_fhog_feats());
            int sbin1, sbin2, gsbin1;
            array2d<matrix<float,31,1> > vhog1, vhog2, gvhog1;
            deserialize(sbin1, sin);
            deserialize(vhog1, sin);
            deserialize(sbin2, sin);
            deserialize(vhog2, sin);
            sin.str(get_decoded_string_fhog_grayscale());
            deserialize(gsbin1, sin);
            deserialize(gvhog1, sin);


            // make sure the feature extractor always outputs the same answer
            dlog << LINFO << "1";
            test_fhog_planar(img, sbin1, vhog1);
            dlog << LINFO << "2";
            test_fhog_planar(img, sbin2, vhog2);
            dlog << LINFO << "3";
            test_fhog_planar(gimg, gsbin1, gvhog1);
            dlog << LINFO << "4";
            test_fhog_interlaced(img, sbin1, vhog1);
            dlog << LINFO << "5";
            test_fhog_interlaced(img, sbin2, vhog2);
            dlog << LINFO << "6";
            test_fhog_interlaced(gimg, gsbin1, gvhog1);

        }

        // This function returns the contents of the file 'face.dng'
        const std::string get_decoded_string_face_dng()
        {
            dlib::base64 base64_coder;
            dlib::compress_stream::kernel_1ea compressor;
            std::ostringstream sout;
            std::istringstream sin;

            // The base64 encoded data from the file 'face.dng' we want to decode and return.
            sout << "RFYXmMpdiStV6dVZSJkJX8t7GVavYwTD+Fn11ZjivhnFQyvVJ5t39yJYrK6Qh6K58ovzMlgPiBLV";
            sout << "nd+ZR0JYVCVvwRapp+dznB9SG9dJbBrsYH68k04uOs9ehP8aK4/EXvcZG6s+rL0rnhVAf7SxL7PT";
            sout << "r/11jIuASMa1daKZjAm5Sc1icGXG2FJjO6CxM8mOzWJ1ze69MPD1bz/QYAWtMUqUUIAM0qOPHY0x";
            sout << "T8tdU+Vo6S6E+8dJpV6a6iDdocbp91meDQcT0/kadhC2tmn0eZoNulTn5MtmsmEeuPI2lLLcRJ9P";
            sout << "yt3c/OJIzI8FaDzYG6aWJ/yBQx/DJF0avAlh7V1UmbD8O/dMoF9nUFDwnhGyS6DYfTXxCYgVgoj+";
            sout << "Ik5RLHY0U/DhNTciFaLX41/MyIt0xcGtxhoVcvwkfIigKnYQsYfNpRdUWseRlZ1KYaR4Oc5B2tie";
            sout << "kH3e5AhrY/HtffCah0sf6MBWJEi7CH9AnVLDQefL8Ph+qCWJGf7cGnM/oAaHQCzHIHVi+mK6EBnN";
            sout << "1NDrzbdXmikwYneB3LUZxCLKZmxsFduB2HgiS0A+tTK6IYc+jqCHqz8N6Gw0sSjAK7rrPDTvxhSN";
            sout << "lX3f6E2IDfVmyvk0l3RhuA1PNEh/nlKR+YxcXHyYW4wGf+UfWScAzKGxrHLxLC7LQycCEaCMkU92";
            sout << "SQV5NSSlwKYKACabK6UJ3gGIpvuQK2Aw7VWmC0iLczqgWsX0GKJR0FAcVL9Ed3nV0Wd0s5BkjBsr";
            sout << "RbUKzw11Qu0toj6BNfwXo/5cY2dtjj93a+CBfNrSEuFyJzZU7cn890c9m+q8C41p+wQdf4pFpjcV";
            sout << "8Kz40Fyt8KtxItWSsACIwmUO9h7DGnyGskWBYrxgDV2VVlvuPAnnSCFPkbdsa/pfnohUq0C5a/ii";
            sout << "BjASduPdaBHpjZ64f+TIaXNAGdrFiN61W6e3fOx4fLFlzPQ8szyWuuDh2hIz1FMflbmu6UOEkQji";
            sout << "w+bwDDJ5OUFmY/00+3B0XAFmj7Pt8OQ70lAVLcX5fC553diQJzrrlJ5p9/8+ILln+oleUVJhtp2q";
            sout << "VCZ9XknXLjkQik30M7orOj+tZt7HDgC5sz/wHU5arOL3nIX5IuIHBJRlB8dERZgoPNQlB090rItP";
            sout << "MuT+Hyr/eR7Kcux7Fy2CoMxcIfWEXvxQoolLKC66q4+SFirdMRjXuwbRXrUBbenmBfMMNDAOkQKO";
            sout << "Bi7d8t1wI9ulNbACtqLbmPjW6iabc0yM4g69cZjRx/JYhV5AaykROJxCP6ZKTw+3ddAht8xoyHLN";
            sout << "40rB40fwEXIvv7qxCdCa3h6l6IRV26fOLdcew1G0qjPORcKK1TPhzmneYvhPZ1m0r6KxWNEnYcFq";
            sout << "WhDGNxj/05eBy2qIiIU/KUPhxKyipF0ekgPoCsWT3S8edYjoaIl5cI0TNpNEwKGRLQeOLBDt+MEh";
            sout << "z1yKm0i2jrtxDBpYf5FW/Fln/XJAK6z9yqSDDTDwQleabvRHDH9bc54bLc0TL9g7/eIj9xcshhaB";
            sout << "zbi7baB/yUnI1+0N6CZ45gV3BcD5n2QLiHME8wELveiMxps6MTf3SdKSRZJcnoVfN4AGYqQV42ec";
            sout << "dFB9y8FLZVL3/8rmB+XEu6YoiGcNK6iATLYQfF0GFRrur0Q6bQgdvXv1uZKtNfYfznsAAu/KBdxX";
            sout << "8qskZBMGA3LxJC3j41VW6Fviy+XUxxcmG9ykbf0COJWDul6ZQ7iRI7rn9EpFIYBM1lKzjdC0UFTW";
            sout << "yDWEE+mf9y+RZdlxHdROFj93FNwzSdzNr1yjqHvZHBZYiuArHEuDPXdxqVRePcID4EHzmpDgWFwR";
            sout << "o5qqDxU8e9UYfS8SG545SPZv69SJVJKld5fQLZ4FbcCjv7wTwrOKTROvurKkxopKt1n69BdDA14H";
            sout << "mViSyK22xK/F7/ydjLoqx6aJ8xyNpoUk6XIeJ5Ei2Lhk84VQk9dxzULVy3KsfRUrZCTTi4YiXkHJ";
            sout << "SmQx4NQKqHR2IOgnJBZuNG9J3Fzv3NKhQpmKL0ZbYLXWdKP9FHWUR0x7y8f74Su+GrplBsjh9NIm";
            sout << "QdaKLa3NvJB1TML1/GNcdJVZUuSaX0cQn4bbumvtcENVbC9u99fGnsaS5FOu4AHd3338zLUMy34C";
            sout << "OpJjU1c/IElgyKmaGYlAolgqGU3lixhxPGBhUlXGfHmST2ZWq/l6NxD//HQXRaRUiQGQzWCvhzOO";
            sout << "ywUlVzl9eJ5e5cdLWvffsPRzBgRMrdHJG4TbSuLAREsSD9QEGab3a6y+qa8T3Si/1Ut+Sn2QvPh2";
            sout << "meqqk9g0fRWtrWxcnbUDU6zMlk36L/o/y5inrHdGY+ixIewhI0n4/Nl3wD96SRITcEVSx6K/BVot";
            sout << "+qIP78I6uk+miUF6MW4AnFyCe1wRNyhT48638KIphSQSKdu7TaBndi2DNgFFvWrm6/cPOqkmCzGC";
            sout << "O22uwHyuY9XhafswKLH02+VD24PIS/Fw7JMP+KzvfCHQd4XxxdsISe0/cjwg26ZfGcnULLY2E+dX";
            sout << "LjdgCxNyFBzFTQ4gB4QExF0dHu+SPoo5T3VAojJbYZqIelFY+u2yQDuS4HCUISPkwuLHXHbcBuwg";
            sout << "5TeuFhyBrlwxHQC/OPACmQJREImiqpzrjmh5QipeEgYHK3Zc72tYSeY7eTzS4jj0eRQ8KiNIGSi2";
            sout << "2LjzAfN2Zm7HGbiBtKZVen96E8HLcrd3nSWnizfaLLWTWB3zu9zz9/vFdaa3TlO6BidYsKomTCgB";
            sout << "wy8yMeykE2qbxgrpRqEqmOkrOI9XtTTJIycfAlwBwoFwuqvGIPtFrYmC/MwRMCphg7acSRcjZg81";
            sout << "5IEqpoq9ca7Zc3s4foteVMCJT1A+qmNAJ/j7IoyeX7GnlM3jsqpYt9BmKfbw5Dr2JB9vzroPV++x";
            sout << "UN2VXRPbahjbIvrTULpeBdmlHU0i3Ya8H/C9RY6c2DhImZ1gDjgn0jQ9GC+CsZpiM2xBvfZZGOEu";
            sout << "c8N8pdo2owD8s5q2G5ZCGNdME/AG+iIlb0P00AX+XR8FYhxKb3y50i1giM41mnkKM/WMGFAnpiuo";
            sout << "YordYSi5plePBnxBfd1Iq46PpsD/n/uUTZMHs6TGp1hM6QriyEhOO261HNHoU+n8m1Omz2cfRJyx";
            sout << "AuFLwHSEqvGCSmslmoDpSg2qOaIWK1LWlN+1sYJj18iL4GRM0A5QzXaS0RThqEgmPjeBOkFBjfSO";
            sout << "hB7mb3sDbY49qbN6P48bGV+yF6y34gYAiVkm2NksHzN4ovwg4O6WMQZwEhNk+4gTIzG69jIm6Hbn";
            sout << "2l48A3CYmn8gcjZw39nrlSxpMf7KPkRsdvGmc5Qx9RjP71zH/KJ2TXP0xxzsaGgmqzXfey5l0Hih";
            sout << "XZtfZw8Y28fHBfm3bnIncS4w9S91no+RYMv0aqc9ty7l+Pa28ELwSgQj9eP4u/i5iq/GPmmSxiTd";
            sout << "Si/eeyK1RFJEP4Tv4f3PkV9Js+azu8BbtU+BLO1FBlVg3CzXH5Pc5FMujLdmlqa495hTmi8YW6Et";
            sout << "Fx8dkC80mYFGpVjS+B6pcQLbLBL9gmKzJf4L94/gXZ25BEDob66+XOaRnJ4RkSAN2g6gFJB9lJDh";
            sout << "rLerp3kP/ubPCvcFywuGx3UjJuwFNHE9m62uiaXFU4m04Kc4n7ccHc6hYUkhkY53v2Qb5SDx2qCf";
            sout << "Yg+PWVXujfYrqxRHSwqtV3yX5kMrtYsYpygb7crweOt58BWUa3duyo23UGJHaCwhGwXat6PEC5DQ";
            sout << "2Oe3LVJmc8eYtD97mHKFPhptBl5u2Bztb3zis/oNj1NdMjnDrNuscEAnrpk1CetvHKLglK63Zo/D";
            sout << "rf6SJcmGR2h9g6wAeV7UdsfD6AvteiPj5sl4UuY9x55pP3CTTYklBO1MaDd/XO3A66uMh95RZVGr";
            sout << "VWDd/uKL+rIuI+vKjz8rt80nv3SyUrY9fbftPdK4pBaVnIt73yZrrqv4Zr28H8XpFFQAV9BPlC9o";
            sout << "a8G+AFx/+W2cSfo9r1Uw7npVvRTe6TtIiKagYUmWpx5BfX0VH/VAW0FUh9oiVfx5rm9eaxfSQnD6";
            sout << "7qBINPxsKq+ZDSXni7qfC3J043Le/uL+3XUqsccvEMoU65akKC3lmw1txoUukv92oxyqPX0eOGsB";
            sout << "AU4JdXCldqjU9K3QhyCvv80ZWotGfUr0TlN1LVZqcF2iq3pX1UDOBsPwz9v0QNg8Bmlqy0Vs+MUj";
            sout << "nMCwU9xErzkXLsuVaG+Llk7mmAl7C34BF9O9qSl2kCmbQYoQ87zS7gm/pK7aKGNsICHrar6vlsKo";
            sout << "BJA++/8XKL3nseNZHzq7hKHnOTzagP52MRf+TPXbTVjQPKnCKVAZJcsOlkmuZc7iDnLn4muHDRjg";
            sout << "y09EYcYlFWhLAgsWmatQBsT028ytgMNrQGHDJdjuNkxYfPo+/91ijaaBiey+DgrUVn0fm20k6/Nm";
            sout << "colrwPwHrK3uOdgBn2ysDeUXU8NLMtR94fIL7etQ9tlUuufwrxEL9zYUM8tpks8HDR51xgTwUOVo";
            sout << "DyGFzOdYQRzwi+kkEPEwkpNQbB258d5w9G5eR00P8B/aSjm+w4FU0MsXM0GgPxnQ+gTpS1cezLTn";
            sout << "eelvJYiq/IInLLxoCXycZFPt3WFQqOBpcs6TV/QucjI/5xMZtP3JHUFv16UKPTFI7p9DF+8Ch5HN";
            sout << "gWXCnRSPdYR4ZRid+Xfzi0TvQsXV6u6PaE+H5MpyNMBWhCwxb6FdiLUW0BswGNpHBaFxjB26Qbmv";
            sout << "OW+s0OuXDvKigjQRkeaYawjRAIAN/+CEYR3oUad2HyJ5Ybr/lRlybQuuIqBhuvpkYzszS7BqrxOh";
            sout << "FJYaivT6r3HbHjaJ+Yz/zNW4KsL80zYkPMP7QgcbbSfE2mAavr+ciXdZBqMMUR50sDNLxep9+hoa";
            sout << "ys9wl75QMdx1jn1qn7f04JMSjCyZ7M4bWSyTW7VEr+NBBLmiMzhI6Ufh1iCUpvrIDSQwSDUL88wt";
            sout << "oSiouRbqizt36TldsvFV6afdLgjRrp2cb4vOQBiltwnY06JraGZnsrb4UCfHZhxd8sq/invK9tUd";
            sout << "D3z8hYyLGbS+3LBCK85r74IYvCuhoUp+KobIZPhvWuvdjmmq3SAxIKHNdLC5hnLVMhGJUrckc18H";
            sout << "9zK53uB3QXX6zGKK62Jph4aOdJoDQaPL0K/yHgn9UayEhH/N1uj3Ao39c05puaxzcSotfBeS5+6K";
            sout << "WYyOOMtt5ikKz79qfj6dVWge22fxXUc6yHYfdga0IbYWRocIx+DuyUZnrRQHihNKgYpvF2vhCX/o";
            sout << "R097oHI4ojZFAX/ZWJ7igJvX7ChiwTjK8KDk+vJ4SUd3IHXaiLkkkd9p6tCuc9Lw5jqWiGrrQKuI";
            sout << "7AmGsPFU2EsfOzmwdZctDGXq2/IutVDmwGiucpBKsRN4y12Q1FWKpceVj2q761LfDx2qJoeZKTPZ";
            sout << "jHPdXnGKcWy+DM6GoH9e5jP4CW+HfdHe474bHfLDbP4NE1oF1vdh4NcLy6woi47hg3FS60z+wePD";
            sout << "bWq29WsSwU5oXq58nMxKOBiMcbGFrkOme/Z59Ybi7Cw1+U3nGE3evCFyVMC6g4f/jvCyWF5I3Nm3";
            sout << "OqmkO6fmZ4ahql6C+RwfdRM8A3FllNPgO5riBNX6RA5xKj+JS+OZUrSSN+tUqcgN18IlmLBExEUt";
            sout << "rdG06PKy+WM8Cju3gtbOFX43H4URr9CQcDxWbN6NoqgF8k5a/4+xf7DilfGJg0E2Vu8GG7tmFSU/";
            sout << "LS6gtfLOFyEnQkTzqK8OhVPVLT62cEfCZN9ZY3iKQyZ+VLQhxwarUAgqeNAMXM40NBJqnUIaaTKa";
            sout << "ryhHefUHazhfVgx2+GikVF9wvMobCvvP1qYONlL9EH+ufuLEw1V35BYIIClbrC2uMrnF0H3QbuJQ";
            sout << "ma69tq8TDPkyDLiaczKuAxUzJoj9reJOYGYTxzP7AQKmmEmEZ2cX6+2klWcRXv23XXN8Ypjjnj+d";
            sout << "fTdzxV4kzcHwOYMsI92tadahezCm9uOR0d8p9IH61QQSlUlJw8tX0TpGkNhZpv23STjQhb+uxzAX";
            sout << "1vYdYbPOenr5vCyhnpp33QezQj9cLhSv1WweplUmZjHcJTkPBdflRA9AuqxDVVnbbofXd4EJDC6k";
            sout << "u7xBoD7EKC+kCEkx7ygj8Gv5GVKbgy1js4gLuYwhJ5aqdNpqm881kkxntfRMluVcdH3IGAUzWR5w";
            sout << "26eq7Je4Ttr1cC/xy452i3pJocbhCqrNUG85RyB5FXHAv6GMvm0rUIa6IyC/kfis+sQsdYkQ8GMQ";
            sout << "wL2s8fdDT6l38N9JbRNwdRv8Xa9QAjwcGNbP2v7tAzM5MyhHW7FImYVAaNAaLbzE8v95zeGpT8Cl";
            sout << "CWronhkcJRab8AKP2UcqAD+mW1hVEAqyDe7oWoZziKa2G5aW2vs/WG0z+NqL1zGvUekDcmJ5L4SK";
            sout << "XgdQgxMb/1k48YqYQZFtQrIqoBbYn4qPeB7i378T5TLcCgB6SldsFdlNzs/czN0doroozh3W+sli";
            sout << "d15Qnv5WMjOinjh0Ybt13wcUzeT2p0ovTtYLoiYAhDeAibydJETLdcozfpXFIJNUSoH6TcLge3tr";
            sout << "0uVP92B1O+n0MibJvLsLUKQ9ueIiHgZb6bUSUixAg89QCDRLCZgkg24DLZ4MMTg7IRfFm2eR2lmJ";
            sout << "Erpe62rf2+JE5JTqU88yn6kLK3bQ9vmaGRZ1NUxibTcdpo4hH91qIldLT+jrdmhrawRjYcRduYGO";
            sout << "WXgjTbgKRTxnqrXwRD4Hl/B1EV6ggYC+jn3LQHaT6bYd1hORmtuLKy9duSHVCNBAZvnto27l+h6g";
            sout << "VbUF8eZasmk+q8Fn83bx7C3eKoHjr6acEUyQxtWVCbmeaMd7h48Mt3Z3r8TyX3DkwQmpClciwpyC";
            sout << "E+pbYEWMZXGOuXPmcHTM/Iky8jNSWyw2lLVQQUzPOJ0v0dtNipYRZqBQCDtSE0JuA3Jo8l00uox2";
            sout << "bH11ErfGplsGZJejPGL8ba90e6xeLwH5oe/GduQ0/Xk4+faqBhy/7TFeexQcFDRCCTrC8+jATm82";
            sout << "vHo+NWJjjDlEI4+F2FOhpRg7MtrrzNP/e+cD++wYeGjkRplbxd5PeyALnjZJ6kghJqFLL3NJ1E4Q";
            sout << "gKmRErg1xWWQzyuDbbXPr5dwwRyZU0gkG0WTwyUy2dFV4KRyn2IbMH6STm+0af96YF0joZzkUroH";
            sout << "ztMN8dWtmQESq6EYQfGlhQzNoBKLXjN3LK4TMWBE+1N5ilXkgv3cnN74RZHdLhEXRJnF29x/DmVQ";
            sout << "qZQ4s31Vk0kqKdQ8tW0rs82+dMMtFv8+P2rYA1GZJQV4P5/TBU36BVetlN+swvULk4XpoqhTTMbx";
            sout << "Oj9tONiyIiJitC3YiCU+G5uL0YETB8nSKrtHRiBD8k7nYj4fbUtSbu9+lKRsVK7kU41mKdBImON1";
            sout << "6Qk0XqAx2DEK4w59khYMRRxOD4u2zZWDVp+Nl7Sd7ihas/vQx5yLXHKmIpjCK3SQYjJz09txIErQ";
            sout << "0wJJZoSxH8efhGsTPuVrbQpGcHLD7bIkWf5kjR9MmOBCmUGgeeGOyi45x0k6Cx+z9oaaTXYcvRtY";
            sout << "M+R8tW5gCLaOPjfbq4QjP6yfYogoaTiSEKOPcMgiOQKrXNiv2ahVBT/lvkm6Q8+IdesGWJtD6xqo";
            sout << "+CC486Du6CFDzAHcnLMk5c3CqDfFGl5Yf68bV4aGm28BaM4vikeKRhm2tULeM7PipfQiI9R9Gy/L";
            sout << "1yZB26qciwCalP4CA2NVjiJut7FZgTF2bO/g0qfvyKsAxMetRTmqALBJi8QvKqAE4i/8gRlTuwgV";
            sout << "x6EGsUPCIcQmD8aJkZgy8+erSAY7MLcnUXu90AC37BLyaOt0tzJKfVRb6cP8wfZHqJneoGSNAA==";

            // Put the data into the istream sin
            sin.str(sout.str());
            sout.str("");

            // Decode the base64 text into its compressed binary form
            base64_coder.decode(sin,sout);
            sin.clear();
            sin.str(sout.str());
            sout.str("");

            // Decompress the data into its original form
            compressor.decompress(sin,sout);

            // Return the decoded and decompressed data
            return sout.str();
        }

    // ------------------------------------------------------------------------------------

        // This function returns the contents of the file 'fhog.feats'
        const std::string get_decoded_string_fhog_feats()
        {
            dlib::base64 base64_coder;
            dlib::compress_stream::kernel_1ea compressor;
            std::ostringstream sout;
            std::istringstream sin;

            // The base64 encoded data from the file 'fhog.feats' we want to decode and return.
            sout << "AXAWQhjCEZzu/QzFZ/MnPBo16Wlz9pMqoReB3hbWAUAVTYPxi+rDtpjtoDRTx996SnNICYjFv97n";
            sout << "u0aU4GWHl5rmAQE3u3yWzmWXq9iokb9SKaCt4faUqEz5YBGBnjQqN6ugx2qGHiaMxW28KtDtD0N1";
            sout << "z+tuOGmH/08j/HsIj40DrB4oRQcmfDgeDpT/TC6z0JdbEAjr+6L+XyQ2YAgJRXZlT1deHl03+tgm";
            sout << "HUsF/eDy1zYrkno3fVuI3xSm3S1yjFstCPRuzmOW8Djv+FrKtZ5ib71RoPV7/tug2G21ChKmrHXE";
            sout << "DqY33pYKGz82eNtUPtMK/Hjjws5oEfGN163nGkkNhK+OypSEaVbluzXR8Qp8XAEeYxfC+n62a/Mq";
            sout << "Ho8/ibgl0aDpt4H92E3Zyks9kfRGp3eSerfL2sM7794ARBNIYrv/C9uV7AMWM7ZZYKVqQYzxw8iH";
            sout << "y8ZmWZxSS/MiKirtcg1HZ4vlZKh/PX4A2Fo1HUR1oA3T7aaxHRErLzwqTHpDMlThGFcOnwRkxMvc";
            sout << "Fvvv4Yj9l+54DuWqM5TIK7oKZNf43GQ+8xxz4jwx3u/qkAvofSi57n9lT4SIBlDc5VQoge8IBZ/f";
            sout << "99JbA4oNOjag8pEWN6qjexmwVPcmnuuwvnJsjTCutpPttpESohdvx9vl+UxZXV0xxSM6kXQfAQ7m";
            sout << "BGUjl0salhKGRnIHVLeiNHnrZcPYrChiYc9V70YQbf+xjDL2cKaRnjWqSPDPoVU67JcCMEdyyy2d";
            sout << "A3eys1667xqBVo4XG43KoL4792nWhagD1YBFZ6bgsyEBoaDdK0HlW3cN7xtQADCMVPYYnPNVot+T";
            sout << "xsLUys7uoxMSwv7n3Pw1qV0ntmTzNgcMA2F7Trq+XwgEXNdeXpPSP1qTDJnuuCJ0GRNcF9igBJJH";
            sout << "oAAuuGqhwpwtb1zCaNh/ZWWoslcqgPKExTda3jeBMNtJKWU8ERRVtgPQCRptcFaNF+6DKQuQtr3H";
            sout << "vZsNvRjhgOmOeWBlF4t+6c8vOvK2TBnPQ2GFFJdHg/bU+yb5VqnZMs4aJ1lSa7Zr2Hsj8ejfpHSR";
            sout << "bf+67aebVVVnzHi1FtwgMB8HinQsM5ABJUKbYeXUXXHvT+oMrZ53d8GlQxQ5OhwwOcK+sfkpgHJv";
            sout << "en5KfKqby7BxpPHXpfaQlcHjnzxIpE08ZHAy7zzn6T2vnm3AxmJ/5Z1PKPKxYvoZzxQbd7Pu4BkF";
            sout << "zlVGiHn4qaj6SiWlNBADgsViE4wIFS4X4rLpy4K3Ze0l1JmESS4uhj1QDsFx+wnnUNe5q1NQQNhn";
            sout << "lgoAGACxT8AeuZZn7BQeZqGphYKN6nDgLkh9uAsk+XqUbF0xvlQVK1p4vJwF5xDEdatez2tcXSGQ";
            sout << "u9gHAE3SrfIzgEt7lXw3V+c5+YNCpHbn5d0hrBSuM1T99RlQiEPtO0Devzfi8Y3kk2/1cUDdOxCp";
            sout << "InVVOS9cFeOhKwkTj2tmsNerQm3OZ3GJD4kE4aNPWvPqMypaiu5jCpIHgauWeUso0ivIsuOHTfnB";
            sout << "WaMZmBBxlLj0w0DZ1sbQb5geRWiEq/Q/S+CfUcUwMo+AOF9kzyEsZA/XLGm/tjuQVPF6G3hwnAPR";
            sout << "b1x6I5S/ryjf90p2BdC8m1BMXsnBxGZGdCf+8wJetts5A5DTrf6N0wuBu/DJgCR1/LtE7Lr4EHYE";
            sout << "mH072kEQyk1XGUIOJayti54yXn1KulhMo20CsjLmtbArQNlcnMBEWDJVUB6bo64OC6LD+3Crgtu4";
            sout << "HSDydiCVdhCiqc+KWwO5LPOzSDyrCrf9k+219y5ZqKLYkD+LiDURH7UT1TW76270V5tBqtP03ob4";
            sout << "GlSHATUCjYkRkRmaEO0bvqz/G1IZsaFL6ZCF/7PCVY6If0Qe7ih87DZggfBXvBaUUgCbZJNYwHBp";
            sout << "KEOjrMEJDWFmo5ojtzZQF6bW9nQgO5n+SfgsN+tIJGT23DE+aezYDplcndtPiCMRMPmKpGgv+18q";
            sout << "KrxoOiWfcPt5ageh9uUJiRvdxDx+M2StVxCefrnAuH0s+/VhL1sI+sBx+H/jOHAiN4t0i3pDDXeL";
            sout << "A64nl3P9EODy1yqNGt8tUyJi+e/lq/1KtZb5HO27gIutmy64Ajd6CD/w9xyEKGpu8/Pqehk2eU2Y";
            sout << "DFVGCsEMM8Z3O5T9wbQyac8F38eOjNIdpXvbfO96b0VAK2D4OM/ivvVZY1rVW7kb+xtnF05pP6rh";
            sout << "K6jv4AhKPu82sLBLSfUik0wgOIZq9EEsh/K+rAA68eJkeII68TKLYYw572OCLU78WeGt7TyArIbc";
            sout << "OVVNUjo77Rax8z0NsPghjEi1/6UdPzz1+mwDze5sN1B8CElTynmINz1jMPqx+UwcZ+MaIUj5g503";
            sout << "1kc47vrUopa93zLYkjOhAiWbs1CWUrFVvfsGoRQxNqcmI1mMEwvV8pJCJqeXvrzrhkIldyrhzN8U";
            sout << "fvMslpGMtRO0Qb+f4wMJvZNSEk8WnIM4uqw319qysC04nVwe8HQn4m7wIw+hCKrfj25N70tev4QV";
            sout << "/CroyV+Y89jGdt/WtsuchLwy+r/3zbhBvA1mpftzFqt8iqfnl3mhmAOq3frkRQARjdnJKnsHp6F4";
            sout << "2f+gaMfGz848cobhJMFKcW18YRGdESS9RQnAezbjy8eepW7RcG0T5WgWI1ZRi0/zvbOAX/zk2Cmr";
            sout << "V+aRkjXcMofep44r93AeA0lHYQeunPxleyjPFSD8RH8Swmu8FnbW1xd4xY7wY70bVK7Tx1zt5HRG";
            sout << "pQCSn5rVYHWgmzZ95zJjNvJmIbHAOyESEa/ujhvKdxkEOSXAgPBfQKJZbeWvAOH37o5ZCcn89gIr";
            sout << "64oqtpsF31f6mc6ZKWH/YQEMR4lBkmyrKQirPwU6dMa3jhMhUcZu63Kk7wL46zCRs2KROGO81B19";
            sout << "AZLwQsGHIyWGKQd4XPTHru14Tfi4IHKKrL+iJstIS3nPonHgZpC6FekWsCpglVVP6zNki5+kVGsH";
            sout << "72qtnX5cKnNaWaJJ85mlmoNu+jW0tKaChqlOyRzOoWWfYRsyMONAs+VYfdV6TtbtUwnJ040DtRSo";
            sout << "+sfcXLRjP2Tq5GteEqKLnHUiUMyCT0V+VOFctHNSLHrlQJ9qlw1nlgbBzf2VcVQYEcd2mevb9r59";
            sout << "ekjCKF380tzz13C3W64VIMkZWd4w+KFnXcevIo4hFqelymT8y42K3ilNM1T/9fbXhyeosKrSnu3h";
            sout << "IqjKEFfkYY/1kTgEewysUEUcN3FZwnwLyFz03McknwFuY6eo9i4sjIkkhYuftOGZdBi5R4CdmEI5";
            sout << "jAnD576o+d5M9Bzb92cBUnBffghYUbU3dk45PJxiXC6pghi6yCO2TbYqATZZ3UeiRCir7kL5Qv6z";
            sout << "Hd+BUNA4QwEYUW/FO2Gcd3xp4ghgULVOjaJW2YYSuOE/NsiOxQ6gjJakBRTnFBwsaVjPqpUlnDRT";
            sout << "QvyACe/rPw4uCdg77C4jOcz1ZHtpC7dphBuILcSKKUGFLee8TDfnRHodABnyYx6R4HawKuq70ruZ";
            sout << "2fH2MuDViMyE58XfHm4+LEFsq5XVW9F00+lOr/GTa0xjwF+D9NIo53s9FxEU5R8wK0Vqk9/UiFu1";
            sout << "kXlGB1vSzo+Lj+XeMHjetsZjnKnfc6NIalH98WgHnF7qevgdwauusXOriKCVnj4k9pY/CLAMXnpc";
            sout << "QjIJMG86e3zPiEla8ij39PdR6AlM4X2lEmuSw0WNwSP7FeiCmsC7Foq0zVgWxmDu76sWg9Gr02dJ";
            sout << "vqM0toHjlCWXVsdqzSpQ1B+F8SQ4P6cX52AakgaDZn7ByWWX3C3jWZSF4trEG6PNAY+33TrQtcrQ";
            sout << "uuPzaykDkx9ektWvu/apPn9kQvIsTje1pb3xkXFnAEd++r1iwBTlNsMvDOZelxYdpJwXamhAiGVp";
            sout << "rzZNegPbfdhzJYYyo0NVfL5hDuiFe571YgVs9AYuC3DYRaBy5NUNiFBo26kEJ9juy4XluFmIYXvy";
            sout << "FVYKJw4Q3kRAANGEXuXQYQ8NaMYBOUyjTdO6Hoj514v10qkvcjdlqoK7Pho4dU7G4q7BbNG2KWUC";
            sout << "9qCwYdWkW6PSMMHpSN5YdVrzKXRb4ZTdSU9zPSikOs5xD39RV8qJnp0tmoNrHCqq7Z86SArz2x4O";
            sout << "YW0c7Ylzo7chUXbe6zDE2kgCvATh+oNu2GjAcPQz4kH5lbkt/HvmIR6CNLg2uOoXi/Js2z99XsQc";
            sout << "RfeJAbCdqNZ81SzucSZdEfM2f0hpbgkbyf1K/8/Ct7P5hFK5GMd9xCBt2XboD3sJP+Fo4sxpbx96";
            sout << "HbZb0AodP5TF7efwz5hFIf9rGBn8570WTOPdwimQTlwY9VcQumUoP+14ZmeeSTpwSKajoZeM0+Qc";
            sout << "qhZ2mIq7LyPlN6PY1r7Lfxf0B4L8qxx227ycszsHJfZOVmEfNVN1gJqdbTySHHzuuDSqRxklk5D8";
            sout << "x7MYs4zpmWMm5B5JO/E8E4CgTI2bAMMAzuZOKoljYZnkCRvjlGTcIn9Wlye1pwR8aFGU6bvzRgZj";
            sout << "xSQsgHnzYwMfq67LLYsacX0rdXhL2TJhEVPZNm0D/Z/c/t2ZRLdGS1R9v4AyVvsBVwCCdzsHoLtS";
            sout << "TbecO917etchILzqBfCk9X+JVbaxz9A/8L8vdhVsNR55+MqkAwh2cOoLucY013HTR4LAoSiPI0jM";
            sout << "jEQduDk4jiF5YuAnfGdeNBWnOWltY+kYfkXC1xx6K3u6krNwXK8J7yvHibVGnpkPlZfHc7DnsaAD";
            sout << "gN6fyhqEX5cuh6Iah2/cE8bcIgO7Hlev9kFyxq+wCeV4w6tEfW9dYGVDK/Fpv2FGHM+257TIWVuP";
            sout << "hLfsGeYXy5BoiZO2D6tQ9NiQEmC6kkpiFLg13Ied9Gh2qlHzqfWhs4PHL2ClT79MBh2ETomUM2Zs";
            sout << "cDcLSLIJu3HB2cqL9kR0j026P01HXFgN6H/ojVD1tYkw16MVjzbQsDXhHsgHUE74Of2yqxRB9jG8";
            sout << "NFRRIfIym2fbjJWmf5zPuN7+i7MlhFZBvIXjnSGSjB8EUvWCp8eWFoaEQakMoCXWbBHP6W7PKj9O";
            sout << "GPcBp6hkThOiCA+28mEKsh4CrN1QBLqVirkVOxu/iAOdAJB2yAggrmLIaPoFle9r6kZjYvvIrFra";
            sout << "9DLLFhTfA+C3d6whSch/S76xIpQU4c1iOC7/sed8h/KWOYwx0/Q7+D1sFdwwrayZ0QwaJpPFOZ/b";
            sout << "SouLRFGblGeWMMTrAE+j/lP9I9mAgl04kP7CvXK2+TUBDIPojX52sjenafLxFJ11ZE9fcqbQ8YYW";
            sout << "4rcHzZzY5FgmlYHNgWrdKrEP4zyFz7eTBqftSKDy+LEUeHCgpyPcjJRlfWACbN1UKUGGCddXwvTJ";
            sout << "yKtFA0LXYWGFK75vDmDL+Xg6jzgo4Rf082Te8RZI7qdqMGOsA8YvPNfxqEeg1St3j+NaPuR+XLeW";
            sout << "2WBCWWMLuGaZbeSdFvtaYYewfD6d+KDej/6UePaaIY35dmsYpLPTZUrUYnkicb5J3+SEjOwuuJNf";
            sout << "Gd/kOL2VRtKvBIcuHIJQpt2r7HxZ8Bllxtb3ZhapH+eE+AY+URuQLSgFrkji440gh+U86qfs7XhI";
            sout << "PfSsQ6deT9M+U0GpsGIlHsvUVp+NnLlgtR9pd2IE9QfAOUEMY6ELZrhvRNQm45adIIsstjXxWUjI";
            sout << "mopgQyqLIS5iM6ALMuM1zCst5HyL+ULMVDzj4X3Xz3B1AOfMG8julIU85bcqrLLJ+ubxdqxrjU9g";
            sout << "OgSAp1xxtpTQ1gYAS7GowFKqR0fXs/jMmo/Ji/+eua7RMFZ1ogMcTdLvgKvSKDOeSI/xamcNNHS9";
            sout << "3N/wikxaYG6OdOSAOyTOlGUFLSj2o1jcAFsJ+TGaV1fpv+77DmUYdntgi+PvUKMfRcSIv8vx/YKy";
            sout << "OHyw/9F46Fhmd5bOkVfEQa+2UdSJkTmsFBTKtx0r6DbCeP47PYsCONkZS1BxrPz0gxUixWS5aUeE";
            sout << "BIANqE/s8LC0mpKAoq5ckjcBKmhuqk//s4FnYxU3k9G2QoC1ZXrxgTtECQ+JwUK4q0gjh3WnSIf9";
            sout << "6Wg0hnha2mu2jvaUEpcwS0b8rHOqyFUeP6dBSL6xh9boWqCFzWHo/pZjqCNUk1bQJIHPlZMV95/B";
            sout << "XqG+vg6P5Sii8QuaHkXi9kJ45bhUFd2HJF7GOk3RfqcLjowQWzN188cLSCH3O+S+jFVIQIJGWeS1";
            sout << "IgXDmvWqPtlzA8scAuhwpFg4gOOKSddQyDQwthCRch83W9SH/1d8gQ5B/2zt1NgUYad+tI2F2XX/";
            sout << "xbeiDa/voE0ooKiMXzYQyX00Zdx1hK+K1wlhH3hIArY00+BcEeMN6Acgme5j/nkeDBCZFdtNJNd1";
            sout << "owsBVbyQHFqSIuTo0dSBu4akLk/xa9sq0bFpBf7MoO/svYI+x7FLIXcgJiEHXPxPzYAwc9UKmsGr";
            sout << "HKEuFXBEPsuInRhtTKYYHj6xcFFfoP9ACtb8xuMCG2EVzziYBzyrMXflrDT3z5bFFMjXA63RYKg0";
            sout << "HkO4FtzA3g7Dk2fgy6+WBkijFqXOkSctWRQ851pxt1ULoqgcNFCJLEbJDD/5eRQlTlqrCkAL8fXQ";
            sout << "XHzK52EHiwtC3whD6XL+iSCwvFwLBLLTP0tyC7+CPFVHlMsKAN2yU4rapQKqttlVyRumV4JTGGWX";
            sout << "BrxvvEXxdlm1viO/F8S1zLGc2YzzKl4XAgP1WtHf8rxA8HTzPokI3SYMraj9ZdtETnepD24uvdJu";
            sout << "aa1l+3Dd8iVxa5cvo88RGch9Gwwwz3AYe5roltQdnRbFnWiTINcdr3wq5LzPImEAuClOvvEoJ26P";
            sout << "F2r4uP+Jq/sL+Q3Ta9+mW2xsUNuisO1F6w7kREQMeWLq7DqKmY/OA/+DFd+yzsTS+WADumn8wmPe";
            sout << "8+legB51oRGRkiJ+Z9G94i9ErrzJKdQby7ynlXJGp1C9Jg9GGkgemOro6wdEHJaA7bKTgpjMHfVu";
            sout << "o3A8IKal+GFdtjxRdcNHZ3P45+t4AmZ0BVi2Y1WSTmFu0elXAaQASNxOs1Z2y8Fi4d+Oj77+Uucl";
            sout << "br2K8R9N1RpxSdKyx1ypxcBI3OArTicmRYyruWR76ixZH7FDRK0VoCjfNc97UjNMvkMRh8t0T6YO";
            sout << "+J0COw6RnbeJRMw29xqZSRkGdUHgTba1L3TJyASKVrxKrn30LthHWHh4O9wHukCBoEc9quqFbs/9";
            sout << "drIlftej5pNPqTjiX4GG0nRPSR9sylJxRaYRCFcuJ+hwfMS7hYFGuLIetwPAT6xRBreDT2nxdwEV";
            sout << "yEXPAxDwtRywWlQfbLsBcQTd2Ekd0hiD6ZS8z2vXkhfwaJzBjPgW2ouPPdpBu2rGNT3ByBd8KxLS";
            sout << "blF93hMEATMsIW8xCbyBK0jNKEvlOQ1TaqMAgBO0Q0jgySu4OFXdhNRs+Fdzke0cIgpCegFyg4ZZ";
            sout << "Hb8EeVGH+oRU7WxugCaJdKlRQqwajwAXxIcpbMb4GPsNamkRHcpGoylHOIrUnK21wInc6n2mAz5g";
            sout << "RF5r1Tdzc1jk3JlOC+27nPDLVqnlEFcOykXXQHXr1ExFeoIA+GS/ZKjDKLUaWyA0aHVnBg9XnQt1";
            sout << "ktlWLzPfsgkuUvrevRnw5jFUnuafFISjRqB5UYb+YoH7E5jMZubLTkNASl4EchktW3dyRFqLs1tD";
            sout << "9RwlkvgxnyAvyHgYNHt5Fo9hDjRGhCzpAS4Fp8tm4qR/11scZqt71ZaWB0KPbYT8DllohNQJ75mc";
            sout << "AzJP6r/Ci3eO4JCCJPzTaitXbSogtWc9TagPuTteryXK0ixsELmNfRXOhgwIn+IfVqBGB957zLVf";
            sout << "yE7wgCCfKKdla35C5PD++7+xjKtJfxNdnvpZASiOvpaAVodGxmA6DgOkXxCpMxVPE9PjnptD/fSZ";
            sout << "Or/15eXLGL2TbCbGkTbl4aQjLJaL2bxXUpbDhW2GrJ/9qR2odKjULqnPyR+ZZ4NO/KJb9oa3aJ90";
            sout << "4Q69Pd+j++w6WDfF6CSfthoNOcAUu92MhMV70MPFiH+7MTKAvcxC7HmsdMgPpqCkKFql4KQ1eh9T";
            sout << "q/f6BQ7dfDu+G/XMKhyeiaVRxD9uVa/lKGxunlY1v9q6108RHK2bsKFPfaUXCuSXeoeKllVsQ4kz";
            sout << "S3hNTyBrxamT6IwvgeE9OyKY4SAysmf3fDAFa79gQTLStwckaFh5rVsRuE0xwKFKWPAYVGl8ui2B";
            sout << "KVmX8wuBUEba/4JylDtwiyBxT17d6+TifDa+zUo6KiM7Nh6Lhe3HgWI3fBaV4SEbfG6riJBd7Zkn";
            sout << "pTmKm8+whfUbjt0gE4SmSe6EDXmKAc58uFWa7yKrXgdqlDlug2NYCcjYyINdGZmFB4mVDXa5Z/gc";
            sout << "ZYWhG4zoM94hwVEBC9yc4ZZS3od77c3JW1s8CAlkBgLXs9xyPdygavcEqS0InxrVS/m8t0HGqmXl";
            sout << "+/ShHGMO+xkuUHZ7h/xUwXeOrvb/slYIKzIauHZ5p7z6VC8loIVZSpmMG3s1OQe3YHthOFMEl6N1";
            sout << "v3FsseSY5Pu74fSmGIurGr0PYQXuXeYI7vh2qzQ5Clv5X75Eat6LwBW9FpcGIsao1m5NppQU98in";
            sout << "wvYq6SZUyVZBfIXyu3eW1QGrizWqwuPghbWcF89ze6VMf3cGitUdYpk40aCd/D5zs6+fwDM4TvOr";
            sout << "JnDaSAU2dkeDLh1M3cckMG8qI4tqS1Y0MPBRBn7jEC7n3XRT4gygL78l9VvlOslPNlT1M1DNRsJL";
            sout << "Yf5n7hd6YPHJ6Pf0FsyALy7tIrF0Tp2O0Jx7EtcN9qkjkXcnRKXfUBTkBdrX527yrlr72AiCu7RS";
            sout << "x8YeZff8LluAjzyrvVFGMglZ+zWXTmOeiDYtRqvCVx7P+3Sh8R2B0MetiHkMCjHZUbZAfl7IN+ad";
            sout << "coAqbiqLNRcOk3t9y3FIogJoglMZarYLAzWuufRUgCBTv0MbambaQ7OP7c8kGmhB7v1YSHwUY7az";
            sout << "Dq9EviXwZDg7GVpSHpuNzD/CXYblZtf6X2vFiGAstXY9NAR6XqUayH/vlnQxt1g1OY553UWcHNyQ";
            sout << "tVfjgEBVDOBeJs1XOXvXd9hDdjEJUSyYfL0W+0FgcuKaY5f7V3ZItjPBoG4BP+CEFS68GblhibDQ";
            sout << "9HmNFLk/RIlxYoBiPyDnRboeHb19hSOumSLgC8JRNhz/NFCkvPV0oglyNQO3//8pLD7Ph6cx+CuV";
            sout << "sY6m3jX4TegOc8u1cpjw4PLas5lRi9mmRZGAl99cPR7ilfvQ9BnDI1lgfgl7yVLi3cXBnmCY8jgD";
            sout << "iz+pUyHD4+coABQcWq/HYtYaU9hGiLwVKgtUVBtBEXEuMBjyc3JilVJkU1eOvKgijc7zyH/7nQ4a";
            sout << "6Oyn4qLQI4PPqjetVpDIyXZHlH5If1X3SnRBnCQmffKKDRbaT1i5b1ifgBtYs9IsDkV2zqv/tNlF";
            sout << "pIzEFgM8TVQSguxNp9JY9Oz0sErDmq4ZxpYXcyopq8znlWrPC4IlX56ZxoSklrZzm40TpMFtYFsX";
            sout << "gzUV+sYvcVj26XbPV1O78VN+1PFDQCme33vJJUiOqEgRJUtcDVT2jni1D7oA6ZxuEdg5D5W9RhLf";
            sout << "V2ocYZeXaHdJd1KNBSYE5n56wSJMliLi8MCoeXHZtTYSb7TYwgQpdT5ecLnivZXniuxzjHF0uE14";
            sout << "xwiJ1kLV37TvZfnE9/ovT+p+qhSUXUTPLplvc9Y86ViZ7tbD13LSrPln25v6vo5u7REaFkxJ5XHB";
            sout << "Zud6hoFFw0GkKFtp235V6H03sh+hb0+xzodaEK3KizO1iesXapyNecjbPVQw+kmX/fHoo3KoBayB";
            sout << "WSGgq5ufWVMBp5yIRYw0Fl2SxlohR3Py3shqlZ7OqOfBXwercJXosYllno9u/H/CJbH12j3KNpWq";
            sout << "L9+L9zB0KzEbZcVggJRj/urdDJzWiSUQjG7JSI0mMUox+2Mecf5FncLD/CJzGjec90KHhZl+OuX9";
            sout << "alZ40ST6gNHOOnHQ02zb+BxUZQGa3olwWYuKaUwjNMEB4KGAIWEi4JWuqStE9iqNTp2cxeg8QmZ9";
            sout << "Djvzm4NvEekg8sUTHxwb9BG2cIHNK7l0IGjMNqY4rOt9Lr2/C7I7ZDr3tLB+mChYfRMo/jV8WBQH";
            sout << "dFUDZ92GW8DvP0RpRWaDWs4wGYAApwok208r0GW+SH8fiqbWxlj/FUjo3Uv4mkxejC6klkB58AdM";
            sout << "Tkpe0oEoXEJhmCpHukrGyJukSNOU7TsB8wosS0V0xfYI3yGziJpRMnJ9mPYIYDxkHng3fQ7OzdjG";
            sout << "bNCsiFoCXAwmkAAVYjTVeqCIxe+GMdcIzse3pdfvvSvtZnHmg2lnBpT+tn1aAr709hQzeXZDwofx";
            sout << "x6N4sR8Qx63af7bB8XriWvU8pe6h/T0SAHsDUODiZ/hw1R4wQz93F7UON4LSrbBh820wb5URb26M";
            sout << "TTiL9LPRo4LQyVKlLrO2CJM3g0q95Rjt9lv7cavqDD1yB66b9Lqq7XrrFbkzL7CObOGsoMrSL5jI";
            sout << "/UdEufYcTbzOvQ64H1kg4mgUWP4o322hlgnoEmIFdJNTUMqACcc1znE+cVsRvEeJzQEhCdlX9HhN";
            sout << "msmDxzKuyLptynyS4Aq2IPpO6E2ESKkpJvpv71ErGomQ4T8+Ulo/pXo/IBGuPhH8Yzbn3rXkEsYQ";
            sout << "C6QmmG5ncTly9FH26brVNOz3Eevxnlg0zvKe+t9RQ1P2qRzP/FCKHJh6D2E7qJpKIzibTE5EM2LA";
            sout << "glUjCsXRazfXin953/EqmwGtjTYSVgce3wZet4dj35W8uONQelH8loZrlQ/TNF70rb+MRMNvjnsR";
            sout << "JMn6jzn5jGWNZjnjg6ESbszjGm+sMF4MPf4yA0U3ZpAJb4f7w0puOv2HXLzuJZep93es1sB+CHmP";
            sout << "IeDmRR61SUg2VWFJQVVP/T/LNSEbO0WtZTZVmVoWsfG/BMkvnRTaOE1GvFnL6yK1DOIlymchqLf2";
            sout << "SIA39vVfiAHWxGfNEBsqz6i7qmkoYu+dQPYxd4FLP3nE1V6pM2cWug9OYRIHnbl/21Tvyxf+mtTj";
            sout << "Gq+q1BAynJA0RSFEGZwWwOHC69cn0fTNZqPFLttA0BI7NYgy8jcNS7WHgcLBi1U793+zynK4GUZI";
            sout << "P5hsoUviKgJTqXP4/QUsApzc2sGSFY0XRUh2bslmDt6Dx9EWYqOSa+e5e2ldwYWHdgP4d5xUdf99";
            sout << "RJjzOmZbCAgmPAJyDGgkBxX92AB4mBKKd4xm15LezKQuIAhjgeD/Hyi5LCfcAp4H64J2EUDDua2c";
            sout << "GlNCX2phWbsWYZpPiiE3DANdvO0vMk6h+kZwtscK2AOUmQ9+WiDTdb9T9VI1crAKfKtaAgmu5G1P";
            sout << "669ealOQ9B94stpNSW9T4SyZbAl9E+5LlAIHymqa4xAxKGEM8ITp8G5YydSAA0bI2min7copMDuG";
            sout << "J3syAWEk7MST/UBGW2+gGS6Caug6LB/Ix49VDMsve2G514J+5L3I/piwDvKQsturbZNPpRlwTZNW";
            sout << "nNCjtB3HQx2dkfOOFiRT2F3NgBjyJ2ieKuwg7aZNzLBWpneAZbuj6SlkMLPfGDfdDhvpyjL5BEJu";
            sout << "0+UJFrIChqmTfwftMDWMU9Ry3Amr0fYPjLK4V+yeyNYU5iYyoKmNJZUfMI0gRSGyq2Mgs4T0ww7L";
            sout << "ReQwLPZ/tet22iRgBmLcKamzpwl1Qc4mX92PH3ug//ktxlBQV/G/ZRuO1X5e2iSw1OyOs2LyrX9Z";
            sout << "5tx+kndD983a9xgELZHcgoviTfseC58c4NlTNNbCE/SRi/z23MqZNSgxGcz6kEdUrwcktTsU1wBj";
            sout << "4nCNIStsZ/tG7nMh7e3C5crp3ULLGdzTX4lJRL3N+BJc4ALT5Yb2yZYtJl0lvyguB+0VBbwm0tuo";
            sout << "6bBgPjQNINcwZKrrJjAENNAjeG6vu1sIE/BDF4XQjIfhwKB7SbNj1iUR5Ql6o4SZ1YFTUrwE0rtj";
            sout << "qKZ/wrJC1uf7PROSEEQqBQ/wOyGQf9fBESkRrSWEvZZVuvXOuA9ONXIS+U7obGZhCWzje31Pi6HB";
            sout << "5b1LTLbbNiWcjUxQcxbyS8vkoqGPUbykm8SDJsZbKP7CoEvzEdWye83cJRCM0Opiy+CyCLcNTC0N";
            sout << "vIKPyySOWEhIV07afoInQIo/GolWvQ5KMK09YRcQIjUFeKSj7ZYU7/cVHRa3KRTuZMaSlCe8gAxF";
            sout << "9eFRGd8xwkaJKx5XS4NEImKme4NmkBtfu3+UaYCwzV6V985yLl/Q+0Ej9D54698/sg0xjsOGrbKw";
            sout << "es2pDYZJ79E33JOWydGY1dAxOlEKY8k+F0NWej+KmWS9sRvkhfhhFAVgnJO1kBC1EXc+3vMFuEP5";
            sout << "+ozf9eIbP2o+fDn0c8b8ODKlAvOwqQAx43XM9Bve/6httRDnYetNOHwWlwHtx78iXIVzOFIfdNdi";
            sout << "CcIltCDaLlddMAgNb069O8yIRQ7Ba06D8iteMOVro70DI30of8bZMI6PKTJLVDdlxnspJ36AQVYT";
            sout << "wtdy1O75DJutxHCGve73ZqUIUa6ON2r8Y7Q/IpVLwjM6Y3L2g8z6C3ncXVlksNoxx9tHxDsyomVT";
            sout << "SVN7HQMKGUzG0UgvhRKxk+pqDNJ9rwHo0shMARFBGTI/6yniW77B/6osIZzThW2SkpWOniDxVoFG";
            sout << "Nd0R+OX245vEBhRRiymBKAMft9bt9QrJ12W57WufXVKGUNEgq5uiPBjAFJ/K85XaVYQHI+JlYpeR";
            sout << "h0o8Qi67e/0jsko/5/HiGZWbncAxjLr7Zv2HSpxJSOyD9OR+7shEFu2s+94OKxrv90POmaKDpPGQ";
            sout << "PIq9nS+MOvvqhOcuaS/XrvsLLwwLG6TsS9OzZ4ttWaqIU8f8FXdIQxpRLHsBOwxrhG3MV+7R0scg";
            sout << "VJiwjhM6HhD4XkOzy6Y8ZCA20UKIuMy9A4SlYT41C2j7Pd0kB7b9gWs16iW2OFvO9hS0HRX1ji3T";
            sout << "wjY7yZ+daZrH4QAKiFPsvg2963gjbtjQMJhFX8fH0ntkBdAVMY0OkB6Py6PSPnZftEzB/kOHuRi1";
            sout << "KetZKu3AYjvWB8xtkN5Y8R+7dz49RYXSnqxWrNkuQj5ZNtqVXGkCc6nT6uF/zq1AjZhT+m0y/D5f";
            sout << "6x18o7EdebR3OJbrL5EcJXUtAgOoik+cDPyWr7jgP5gSunWrKqh0k160KK2aklzmDon1IcnNcIe3";
            sout << "dKfwirPUvDjtcsPDEmiFeesPnuFQAUvSH8tAGlMIhtEZHe6TTPOu0rJGZYonpb4mvorCWKZd8nrJ";
            sout << "Dyyp2QzcCneNXr6Y/61h6lO4XDTcQOYtHWvFgU+4wPA/YFjbs8Pg9Xg+ABumT5JgLTNyWdlx36p4";
            sout << "z5RANEDRwgjE2xbsZD3cnU09aA5Y8ziH1Td0/KDY0G0L7BPatf7tXvE0gETAeAGP4rVW/qPugqpp";
            sout << "fQrGyJ22l6dC0GJRMgmIfaGWCs2a7PxMkE7Y0tzVWZWGEpCixF1leGE3Cgc926XBuit1bFVXC1r4";
            sout << "OYl/9q3ci+DzNjALK3ekXF4zTwMilC23s5z3Q1/UMruDq5YABrTI9P9uQhKY2iBE51GjFBTomsY4";
            sout << "BRXGJBPd8kDnMzwwPXEp58qg434bBDUNbtG4Ek6dfuZL4I4H96STiZl9viZuacubD8EAJuOFfTuN";
            sout << "3LTRTdegqCeoZL7whYS00v7B5meYaCMyxr6D44WiyXAQ6jM9Prlrb/+HHgeU04oaGB9AB15ctDHO";
            sout << "geJPGJU7kfrGAJN664PS1Nh5HDSq3cr/0cxyi3IK0GG7Ab/jtPuyPl7CJIb/IwQ9oVZQPfAaZK8P";
            sout << "H+OuMvorr9JU2qovZx0En33XQ95HHzRFmbu3k8bnT8yYSw7ZEzQshi65NzsR2N6QlXjIyadvdaoC";
            sout << "uYV6v/g7KmR1xzgX+nxK7p1y1o8ZsVZyt+K7JbNKvhTWbMHubHXG1tiRrpeyWCksC+K2IlIRPCAN";
            sout << "GCBoHS+AfqKchbJ4scaBFcywtyEZVOeX6c2vIMZ+a3LAlIHHIhfsocwh6DP+qkyWDvluwcom0yQQ";
            sout << "guXWO6U68PzX9DdhMH4qtPXxMf2ssxD4Ij+QIWQ7iaNv3bTi9PUeT7Tr2VFmrAchUUsdQywtiv7W";
            sout << "7m7BGv/sD8LjR31EN7yqQoiyKkBzG+xE1fRRhmGoKoPrTPf2bu4OIxnjUlBoS+SFIgivGWFfaej6";
            sout << "8ld3oHHJRCO26vZnrBMz0yP5UGOlknPM5ksreJ/oj89aEg6cikJY/9JMngnSMfEfM03gcMOrH+Rq";
            sout << "mgid3FyXOxQClyB08Tg6tlfKgEnK5WHoZm4TKIF6bJOAqucCwVzdHpVpRX/Syz5IA7YOZN7XAsiL";
            sout << "ZLHA9EAwQW1B0lXgfc/SiJ9jW7rJMAH8IHxT/JjCAgGHElbpWfL674TNk4JiYpjP/qilse/SV7vW";
            sout << "GzWyXy/7IHRdy+KsdO0yD8OJqdRPb7SEF0RT5cR/U7mCZG/a6h7Og3uXwGanJxGRtetKUwqmzZmG";
            sout << "QpEwxDfK6iVixFIeejX5unTYDm8TqxwtwCjo/LbxEfWM/8DvyYuZDfRS72XI0R5+b2Idd2VoF2p9";
            sout << "dboCgQjoaq7UZI78teF2Y7FTCRX3NkL1aDf0fOzgOOGIhnBwGpJro7euh7FzvQItnBXE8tbR2bez";
            sout << "mDXQvbe4wrNANqlat/uKutj6VsaWfjlZBh6DpUMg7LBacelXG7CQI/lu21t5z6uTzUwK2q2kFihM";
            sout << "O6A6bdpWLNln4h0GmX0A0Qaq6lPufFdlLttdoqce+7G7AmMeUMdRCG9Wb+ygDi5kmMQ/csr9QAyw";
            sout << "H8/HV91uzqjqgOq+Rx/dBez8oddU2TVs5WUwahrYDyR359s1eEvqyKs3/34NXLsm7WfXloXFyCRY";
            sout << "movx24HjK3AwX0qbY/OqUoaEOrZ1ZCRlYqbDLeAXEtM8uFMwbXTU4xxrUT/F45Nf8UhhP3XsCLyL";
            sout << "naRHBV83eZqtqaFS2QOa4jJx0rA3+hr9gK+RIt4JeFdgMMPYOikq25Eyji3IAsO8ybEZS0wE9tD7";
            sout << "pPz6JLrIT2LsB9g1klikn0S253u+gGxQpWy7HblhgpOErFpEsARz2vik8TdntvHEJrEHiIHG6FiR";
            sout << "fwDLrEvz5hUc83x7mFpUfIzKZWRk+Y2Pc3QQJA9u+gsEavDAdwoLCYGBRQpT8EOJIBkzQKW/W+Dv";
            sout << "CB55oUi7S8q146oX091+GridYSSZ88lfRG/pys2NpQMVH8QXlZ4HoJpZWIIUSn53U42OhvpgCxb+";
            sout << "j0su+EFYfp1RIbJnEHJ9DpS1gQ3a8Cer0R46nwxuiOncDr/x/6zrJIoIwQUCbEUcXM18R3G2T7+3";
            sout << "vVWh5j3X/j6N6u+UzS1yh104tHTNRVsdbTJ+0YtirwIP0XLsGiI1RZ2JeoSGBn/5QUB9WUzZGxcR";
            sout << "eCaqrkQkY0c5Iv9OkEFDDO0t6IoV3NGl991pnRg/eEgq57Fa7bFIyOY7JxUfFlYh8h+jkmZJo13c";
            sout << "NpiBBx9d0IwVkbGFxoExpJZYC2GzMGRgqVvvyP7P/EGugTpUP46RffjYVMHLcc23v7vr2Mx3skeY";
            sout << "4NuSb+eaVsLmWDzP6X7zOJ388iQlcLG6/OyHn5r0CauMuELIX255sAO6XeiXnRHnN71tQen4bk+9";
            sout << "6KiaK2Xt4nueQmQ4cQ+49os3v/QYeCqjvL4Prn+mhP60b7pH5hyw7FAiw11bc1x2NCzQ9ujxpLKK";
            sout << "2wEwfO1d9GOaPgPp/OFrvP/R7SF2rkI9W9ErfqEddLH9oqXo7vbdHOgim6Aq3jWSxYUgIzp8d7B+";
            sout << "EN/8E6FSXHV38xN8kAu0RrglxTfXMh2yl2n9QsMdYAoA";

            // Put the data into the istream sin
            sin.str(sout.str());
            sout.str("");

            // Decode the base64 text into its compressed binary form
            base64_coder.decode(sin,sout);
            sin.clear();
            sin.str(sout.str());
            sout.str("");

            // Decompress the data into its original form
            compressor.decompress(sin,sout);

            // Return the decoded and decompressed data
            return sout.str();
        }

    // ------------------------------------------------------------------------------------

        // This function returns the contents of the file 'fhog.feats'
        const std::string get_decoded_string_fhog_grayscale()
        {
            dlib::base64 base64_coder;
            dlib::compress_stream::kernel_1ea compressor;
            std::ostringstream sout;
            std::istringstream sin;

            // The base64 encoded data from the file 'fhog.feats' we want to decode and return.
            sout << "AXE4jk6QRzUCNtyVtAwaCkqQk/DMMSitt4xqzlezKJ67wEFxUpvZ01jt75shIVSZG9SaMoosu0FJ";
            sout << "Vo02DfunGxDF+x7Djay7Kof50MRY0n02cYkeU3pSZtet60FljdoWL6ju9ssY6Ar4aCxYxQg/kCiD";
            sout << "ma4JgZ29GbLvsjVr9E/GXE0U+6P8lnAFxK+6ry8etw5dlCGV4SDmUd0jJZWfEmkaIQjuaxF2hgfP";
            sout << "TLt+kaFr4ofREWjjdVoqCTcqLDAZBOXtFTDLH5GI4NkWnwalQo0eHPuf/YjdfSnwvGrLe13YYtLU";
            sout << "lqPAEoZKSULsy6+oLbZLyaKdFs684hMIx49b8l9XlCx+azR84pRBA1t0wICnTUlHgkb53fUCIZZf";
            sout << "C4hj9sRC5Cvi/3YXnMF9bLu03xpEi1/P/ksugWjHdkEqd/H/3w5bdR6G9qtR0aDZkBQndSp2tw6T";
            sout << "ObmFBDHSSnJCCOBUfREk2DGiHTd7pYVwQAUvmjDn0jIRgRTxil1gfAQ3x7TZI57h3f8aFooRnm2Z";
            sout << "v9hHk+6HPFcLT0HE4Xmsh9fR900yLlv0n1KdyCF2uRWBSI8RZcxW8KS3fGkSl7u36UV5segmZPfs";
            sout << "c8yjKCATKllPNajeD5GAFBgiPc3Obc06G5fW42HIYsbzpyNrVp8WyFEkB5A1y+CpcE5HhcrjA2bh";
            sout << "LI1JRPQR7RAyhCXleodfxbQJt9mBAp+BOBs3B08oMFw1crPRHzNpiac1J1Ek+FoJA45XyoUDj0jt";
            sout << "cTSQ7XTAp798i567sEdLc76fVrThtvMArEKjhta9dY7M5XFtld39znMDUgAhVRAkEbnMKhsUrVNX";
            sout << "6KEKb8fF4zPEVFut0AeIiuxiOUzkPmXO845LW+ZGoelywdK80HpC397o1a+Pll/in8tKDMtX2wLe";
            sout << "WF4KQVQYIOt3Cy7X5c9zu7FIhYA9ZvonlT/3qtdzPeMU38cDX+yP8eos/Jsm1qGeBh+/2p79NxH1";
            sout << "vFpFZEUK9jVwjnYlAQqQWAGzIgzLYHcXlASZZo7Vxsw+Hlea1ig5YVoWBsN17HsyQYdZJg9Ue1p5";
            sout << "O1X5zCLh+RURUjALchLaaRFDB8ox0J8Csnf4YuGl7be3whRzc5p3CU0xQGRvxR9WYsdtBsU6yxrh";
            sout << "TCpyxRDmJX0b6TAu5cigT34THejSKA6BbubLQ5D4G742XWEUD8orjcV8RHvM0pIMfoauQwgDXJIa";
            sout << "MRAs5xThR9rz064dHXUjiRcmcewdSZAH8QF7hlDZUhLqxGhHUQ3qpH9E0FyWhRc7zsqf0bmv7VgK";
            sout << "aV1W+C9msJLik+2naCPQsSnDieDz6rtVcKDp+v91n0/DMnjXfwZkmz1liZsxIx1BOM6G1GE5fwyS";
            sout << "70ghHx0dF9hw0c+oBEGvl/3lJgoxQiRyDcuLFNqX7KpU0o38YW+RZrGUVSYM2ryf6F7/CGyjW+E3";
            sout << "ZTIxOXxIkPKxxl2Y6KGKd0rB1LDUxEP4X1i+n1VbonV9oHwGRqX8V4t+sgClUn+BCotQ7kOR2gKf";
            sout << "tttLJn3xoyz5q71uMDxfafOluxSTVjkeP0CUL1dfh1U0R/EGbmkxJubtBw+aIa4W2zpjdPCYFO/H";
            sout << "iCjogIuGQgap80ZKLTZ3hcaMJplkqJWagKcZ9KsY6yrBwnvT4AyuySVdIJmkCe6rdVLsMQGY+sVI";
            sout << "lqkcBLWqPiB3fUrgl9WC41ZETr/PcJDJbN4NgxVhbmHRfcr2U08qPCvaNn1QdPu3L8us7mr6TcF+";
            sout << "BM0ajyzTgM8X875mcFhfomVS3SOBytdGTDT9wn9yP+/kofWTnvgqKmWF2AEEs3YVagYmK6IjIolY";
            sout << "JV4Ab9mA688iuA2ZsizxLHm3+uIP0mRZjhbdlyjXMDQoB9roOkJsr8flraRasR798icFOZ5SNB7m";
            sout << "G9ICL56v7jlgW/PEpfHK4WCG/S4TEdLEiSS3dRogOfpr8LzIIft6JBJKAlq7ZD5jHY1fvIlbRXIf";
            sout << "4SQhodBMYXmEU9VLiKoNHExuiIQNzuhmgb1jt5iB3AEV7d53bwdz/dIopdHX7d8BuB04N5cpuAvk";
            sout << "+qw4P2USNeljGYM4p0fRlvAv4vQiyYAxV9homMso4G9LivF29FTtFUQspQeTqe8nO7eELaANqAGr";
            sout << "FZDnNp3+fcp0hF76CRY7hivfA1qxcIap/rku1nsisUMoEW4oMuVhuMmRXHiBYzIX4tQj1KMC6m2A";
            sout << "/7lTUUUk5+Uk28NkmeNUt1JhDLfq+39LrBOpIsIGE7ECQtj13mdwiGNXeS4zT0mUgyk8m/rKg/3D";
            sout << "S1NwmpbRPxaxKdwfWgoZb4CUEs4sOBKcyeLopZfygOfiVyVAmSJQQLMXErTybkH0NypF5ebErzs8";
            sout << "+ETjXr2XiPfZxEZn+du7zv/JZgwZGrV8yhvtNhW0hlj8pwVnzHneHg8U00PIF4mDKd/ZUD1AUkDm";
            sout << "rawKMsgPI/L/8f0AhLugK+cOU/3qV6Jq8OnEBgCX5uMQBqge7RDfbODnxsGoC9Xs+sWl3MS8fWo3";
            sout << "sWzCeg3b5HJYnzYSKoYXYwrv5Ptw5E2l4yvquftZH1WVOeaeoycjqZY7x3Rp6M2Yy+LNQBPd8plA";
            sout << "BtMaYlOy1Eu29DyQ+l7NGhJXF+ocHB6QqCABi+wT+J1sUnhZha7oidMLxBorTEYRcSqHQi917lzj";
            sout << "XK6QkM4qW3f0+Lrt1sgdj4mRuxK9UAF+G7vmWvFXhynRAqkrdf6dcvra433ZQQuCxFgK2+2Rur6U";
            sout << "k+iI3ANRFBzTEU7/VbgJb6amli2XKNG3BtUk2WI27Ndk5zfOCIAPGHKkfa7d7MTF9cTuBpmL6Hcy";
            sout << "4AHIMm0ipG93qbBSpJF1RwFc53b2PfOQIw9yj41St+LgtuJFqz0uU0oVRSZSs5eUM8yqkuCVUmVL";
            sout << "wNnC6hiWLvHEzEdxPm23J2claqeF5l+1Q25MRUPIlP4ExxV152EuZRPU003rY6YTS0BiFPPyTDp4";
            sout << "2vHgifdfhWUQFSccyb5vsnu6RAgDw8g1KtYtiZXJliY1eJE+/3P3OWZPX0l7c/7Ec53Yt4jZq2nC";
            sout << "sQQv0JFkW0OWzHqcbpC3E79Uh6B97QIH7QXBFZKUFpkh7HGlenCz+w57Bly7ja7bAqIcBv0fi/jo";
            sout << "3Nk5VkyUCOObwdZ93SpsH0mgJy3RKZwbyXrqmHd4o0JXqZvw2AgcVmWbCxZw1BofCmjGhnaO+qhZ";
            sout << "UG6Rp10IKzgcC/q4hMlnZZ7pv13OrvAJZPtB0Tt5xvOqGfYHd4f7g5KQIWySrEqv//ZP+DoQ5KoF";
            sout << "bYEF1/7auVwWBrzTlqn0zDioXTa54jZyoam+OpbVStfojdGjWoUm3X9Jmo1rXJpE1mayNOv4OZVl";
            sout << "nV0IqmDJPPHyFbuyfypbH1i9o6MwSljL3+szXe9qSC5WIwhNB6Os5cTawCMdcIrH5/BKb8mkPA3t";
            sout << "dZSHd5QVym/a3NYDl54cwbvnTdcL1RCs5WBGHs5pt4qaU3iBOAwtEbgsoAUOR90+slf1GzpsKidC";
            sout << "hmVdd0iMqbhQBfm/tQh9082+3cLDXgP/LpL5Wo/ROzCbqpvFYAnQQ4k1RxGN7Gwrh38IOYk9gOR3";
            sout << "3brmuN1c55DpdMzGlunOvp8mJ+mQLA2NRF8dKgWG2ImHlRvHOmSll6CmXrNAzB+u9+4eadNDUVH/";
            sout << "qvwmrrroyQfT602HgJS6zKJpLmQpuNrHD3DA9yZM3C5AHM8ql7oyFXCdeTpZk8H1WYwXp3D8663z";
            sout << "dPsqHLkvIIq06FT04h377JXPivoUK9vc1zRIgi4RBitJoirImlvy/j/rJ/rBTs5kRU/V+liFrhtq";
            sout << "fBqz+qq4htW52JZlVAdVcHPBcJtiWhfuB8ebzLcG0TijyhWBM2zHq838+qCxy6m0KYsEVz8EnKQo";
            sout << "CrmluDTbsEkVKdU23L8T9i/UTiUZNoK1HerfY0j2EVzXwFBrFhNRd2bYOdZsLIyAt1qDXy3EYcpM";
            sout << "bYzJfbnNo8ffEAY+HHwJ2vsxxL5fsUt91RBY6gHlFDSZW4/HNyI0hdva7Gs4ylxjpfQuphrXGizD";
            sout << "X/esl0EhUdvm7my+d5eHeKWR/RnPYF9WilObYIOeb9d/pwxamx1C8iMSRSuvfuWxo2fBZz68uFJz";
            sout << "X42Ek//JXEZFVWMKCEEL8Lx4EEIZ1RPnrIwxrZEYkF+5mErnAaB7hiA0IIKRZhn1ozLLuhbbWfDr";
            sout << "qPEPAhV/2qZPrnmz8bB9CAQqTdLdwV0Z5cG5DiwNbIwi6AfekTBkx8mpzRLwmIHVKMmg23oYFGTc";
            sout << "qS5ar6pvuvc4nQltHhr0gW5V0LxKPuvXCfXep7GO9GxQsb1B1+mBB9LkbTv4UdoBmLfx9ekEny/q";
            sout << "2jnkXLxWnT6YRE6y+cE6OvHZ1OIZKL7T+kaSJUZLfXg9lnZ915mjd4zx4cIrkcU4r0IlMg8q2TQV";
            sout << "TLml+8hIXaRY9Ol+XvyT6eADmlDZcrjKCzjtDltksno4w3JgW0cpQIUPHfoATpYU/52VC/lDrsjk";
            sout << "jDs8bAmKq2yPW0/8vojyhwxJ6MSz8EZrN15BddFtB/99Wqhw4TR560wbGfaJvnLrTi+bRlRoK36l";
            sout << "F8V/7xJ02RIRxWaAEyO5P/C/0SBXDjB1o9zwPwLCvViCDdi+oontvT+pI0UYpadfqzN0fG1uHoR6";
            sout << "DAVqN/AeBS1meGUtmxoVRpF2XoU0JagKu1gpAQLpirjiaNi6SNnf+2JmG51BN5I1rtHx/qW21t8m";
            sout << "KaC+8p9asjVu5BHB/ZQ+DEUVuOKvVhxIch7NeQqtNgAA";

            // Put the data into the istream sin
            sin.str(sout.str());
            sout.str("");

            // Decode the base64 text into its compressed binary form
            base64_coder.decode(sin,sout);
            sin.clear();
            sin.str(sout.str());
            sout.str("");

            // Decompress the data into its original form
            compressor.decompress(sin,sout);

            // Return the decoded and decompressed data
            return sout.str();
        }

    // ----------------------------------------------------------------------------------------

    };

    fhog_tester a;

// ----------------------------------------------------------------------------------------

}


