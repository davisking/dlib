// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/image_processing.h>
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
    dlib::logger dlog("test.correlation_tracker");


    class correlation_tracker_tester : public tester
    {
    public:
		correlation_tracker_tester(
        ) :
            tester (
                "test_correlation_tracker",       // the command line argument name for this test
                "Run tests on the correlation_tracker functions.", // the command line argument description
                0                     // the number of command line arguments for this test
            )
        {
        }

        void perform_test (
        )
        {
            dlog << LINFO << "perform_test()";

            typedef const std::string(*frame_fn_type)();
            // frames from examples folder
            frame_fn_type frames[] = { &get_decoded_string_frame_000100,
                                       &get_decoded_string_frame_000101,
                                       &get_decoded_string_frame_000102,
                                       &get_decoded_string_frame_000103
                                     };
            // correct tracking rectangles - recorded by successful runs
            drectangle correct_rects[] = {drectangle(74, 67, 111, 152),
                                  drectangle(76.025, 72.634, 112.799, 157.114),
                                  drectangle(78.6849, 78.504, 115.413, 162.88),
                                  drectangle(82.7572, 83.6035, 120.319, 169.895)
                                 };
            // correct update results - recorded by successful runs
            double correct_update_results[] = { 0, 18.3077, 16.8406, 13.1716 };

            correlation_tracker tracker;
            std::istringstream sin(frames[0]());
            array2d<unsigned char> img;
            load_bmp(img, sin);
            tracker.start_track(img, centered_rect(point(93, 110), 38, 86));
            for (unsigned i = 1; i < sizeof(frames) / sizeof(frames[0]); ++i)
            {
                std::istringstream sin(frames[i]());
                load_bmp(img, sin);

                double res = tracker.update(img);
                double correct_res = correct_update_results[i];
                double res_diff = abs(correct_res - res);

                drectangle pos = tracker.get_position();
                drectangle correct_pos = correct_rects[i];
                drectangle pos_intresect = pos.intersect(correct_pos);
                double pos_area = pos.area();
                double intersect_area = pos_intresect.area();
                double rect_confidence = intersect_area / pos_area;
                
                dlog << LINFO << "Frame #" << i << " res: " << res << " correct res: " << correct_res << " pos: " << pos
                    << " correct pos: " << correct_pos << " rect confidence: " << rect_confidence;
                
                // small error possible due to rounding and different optimization options
                DLIB_TEST(res_diff <= 1);
                DLIB_TEST(rect_confidence >= 0.98);
                print_spinner();
            }
        }

    // ------------------------------------------------------------------------------------

        // This function returns the contents of the file 'frame_000100.bmp'
        static const std::string get_decoded_string_frame_000100()
        {
            dlib::base64 base64_coder;
            dlib::compress_stream::kernel_1ea compressor;
            std::ostringstream sout;
            std::istringstream sin;

            // The base64 encoded data from the file '..\..\examples\video_frames\frame_000100.bmp' we want to decode and return.
            sout << "Qld+lmlZhXVX5NDRWIFG1T4+OGdJbkmxAXXdHZEtpDf6knVTlRWyAhgv85Tf11KJZbhKv1dcsKtQ";
            sout << "fX3y/9RxoGwtXxs/KaBv3IrfvBK5WFRWzOM7H11xVin5AjxdzyzgU28fRgEasu0Jvk3SaMBNM1cI";
            sout << "ZAK+MA+qEKwBn+rkuLbptS4sUNNC8PWaqLqNQ577TiMbsEgoa1FGkZX4feP8EfvV7w8H4FvbS+Cl";
            sout << "yJCj0Q0Bx2fqFCDaAVtlm41VIJUFS7AePKmSnVnAlYYOQ35XxWS2sjAlWZHXiEfKmWmSOTt3x9/u";
            sout << "5l78aPB0FvrUgF5RrvyVIOSgsC250JwINl8uulOOsykmhUUjRCw81dAITDr/d2EAYmPyiYNZWbvl";
            sout << "0Iy4f9sOJr0HnaPdCSvojI9/xdQVT3MVjHhu82UtpJ2cuc1MjUooCZdQmTn1mksdAdZtsn78aRMw";
            sout << "P97MKVgFmYMmG+yekaN6+nYFgTXnuEnatHDszyavB3+73iWTYiYs+CDRqEJ7XUkp+F97W/hUuvFr";
            sout << "DwePerItRkWZN8Nmnjx0Jibv2wu37zkFVnFIAydvV9GzaFDd75hUwQE5rjGN8jp4dBMNwcK6l7kt";
            sout << "xLm5hYF2eo9sESGJqR+8uAhGsQZESUpdBvLmf/+28MVfY7Li0lmeq+bz+JxvIa/jF5ptffuTRtZd";
            sout << "WLbJK+EXlfeorbfh+6di5CPp5/50W5zrOepXHeDfrzCIpz6XTCw4k749ajiMk5y5XUAk/gObe1bG";
            sout << "JZeGfMezy6bNcosolf5mmg5yDk07bK0JTLGFvYALT8aZFfUTHR8+47krp43r6XxaalqH3JFy3pYz";
            sout << "3t1IQPC2wz9FiCGBpn+UdULDwpBt0oqbGJYGEHDWgzwv8dv31NEbqez+G+HSEfF7CWs80SH3yYzL";
            sout << "9EjKo7ANmcYWtr8+3/SpUl0Yyzn/7zDL7191rWglr3N06+bB+bdQy7J1yGueGtTsO5XMqfUEQL5n";
            sout << "xQ2FOU9c39cruW0Fp8TDkEKM6gowBRTelXMA1w84+L9Dynheg1iLE8MSrzvA0/TlXoFD7Wt1cC/6";
            sout << "EoYPulG3BRy/3ugzUeiHm4ZWh95YOflbH2p8Ix+5/sBZ+W5xSr/37NRYi1nQ1jFwqPdAHx/OGvfW";
            sout << "BKhz+PF3146JpvuVbq9fTqjqYeM+qcAu1mS+VqDyJTw3IlvRUsixyJu7/eN01m08yumXP2mm4SwC";
            sout << "Q0j4aPKRzl4rRx5PJE4qnKxmrxSa5DavIO6ik+DKR+leGGk29zEqY+UcsqcJQLKWVy3k40wL8ouF";
            sout << "ihEPUISSZDTE/iU/4wntqVXCP9vRNYnrnXVbh6EFFA6nr4dBrj+qnkkpDxrt3tGA0Dml1u68hNhI";
            sout << "aI2bAca+a1TXb9mn18JTJqK6tkoJmE4JP/T5W03PSGYhRDa0ddNEt06eVPuCrSyDwQ2lQVjHiF7N";
            sout << "iHVbLJOQ49//SmXdeId3+sJ4NwRM0jquqURHrBAYwuEImtqZDgJ9Ac7iC5FDMuigc3KJ6Y3MRAI9";
            sout << "irrbcei2XbALqQH4hsHiOvplAaUwqE2f8iaHHQrzSKdNoRpl/0rWgEU4eWcKVl7JmmwLRWqRDg6x";
            sout << "jYDZ9Bu3KAR3llOBPHN6G6VcVx8TfP6qHb7bYh0+lKqIv7qNYUdbVNZLnfBSrgYkrbf8LmgiUAyS";
            sout << "d3JNuR7XsICjOPdEgC5OhKHNT8w/wcZBN1T2svgaokoYxI+dV9t84s5h+W1RgCLTnEu5Lz811RlG";
            sout << "KEAZ+AgqPAk6PneeZs/Ujk1Q8ISZ/K6pL1SBnZN2RaN731UHMYJKK4/yCfdkqGAtFBnS4vvA82tV";
            sout << "uv2SwPvhD2KHvZDCsAVPXFQWVVkzTWKRcSXe55vgnZjF33ziAFILy9xNklmZZ/tDpZa3I6VDkNUW";
            sout << "o5+eLuUcJRi/dlrKnNaHSsjV6jQd0ud+JB+Gv2YfV9XcxFZDPavRGkNRdOTe045rJ7QKTI5OqZ5o";
            sout << "DZ9Olo25TyXVteyt9CwVAlLB2owBoEmni8HFWUS1TiFWmU+DupCJw60bwE9SLhR1KpfUp32Myg2S";
            sout << "6e7PJVndER84KdqRkPFWZyJsBTXJ6U1g273ty7MLFxZYmCp7SwvtXNrPMJ5yCS5VGa7sX8xavNkJ";
            sout << "ade8iiSs0Upqbm71iDbMCp4oBRxa1w9zMWZ0VF+Qg4HVyKdyRmtJzBmRvUmujyYFCQwMtmzCgIu7";
            sout << "mPCbA7EWkoteGnuFIgag9P7kAzEF6pI46gd6UyOOFF44eUytnP1AqCJYHvNexViNZ0FXzgoCzHMs";
            sout << "m6RPhsqJnQjG+KaefJm52DCfLNOPd6+rj4mv5iQeVEyNUtdHuS3ll1svho7IKMF0KXyjajXV2X8i";
            sout << "BaHxZ25zSlogQy//I4qdHx2SlUkiGJC8jpRZ9LVmaCWGCnbEnHn3uiiwTLRiPO0G2qgbMjgMvius";
            sout << "XjOSu1Dlbz4A4XJ5nxRyF+7uG450OJbVF6LUYujHukrHn8LADkKQxY60sddDMcx16lofSVJEF2X8";
            sout << "pwhXS+EcqDHQ5pT0Z7+fMFSumC6kTWJIfcwXrTJDcWslF+cnQw/qmlnSjK1jnTyU5YpjyByETfWc";
            sout << "hpFJbt0INtrUIDjOwXha0dlbC/jLXrVbW26wTJrKmi7ftD6CJSE8cZBGLXCX4zakRRPg3DEHsv8D";
            sout << "BFwPWlX7keygRS8L6o8teh9LHlYSbFqlno7FNqKh1V7D1ERDvsBO+6xoabo+mYXOdtyVWPWxJ0Kg";
            sout << "1qsnFibHgmqi4RFHoSMkAcHVYj9cHhiFo3DSj8sm8l6FGBQSm+o+oehamFW6xIL9ICgm7gcBQH6N";
            sout << "BH3f6q0CorAetVw7kyLqNiEt7G3V1DRjJ7bcn0/ziYsn6dZQ6kq9gslBWZe0dp227DpameI0TliG";
            sout << "pZ+d5+sm6g1jE1H5H6Zlh6jJ9xDMywzNFOitOdcOL0YFuzsxKXVPA1Ye9/wYqtUyJbFvO2ZqDSbC";
            sout << "k2z1UcbCv9KQhonO7aO6qgkaRIFmPmV82R7JsHqjA7CPQC/hURfB7qWP5mzvyB8oEqtd8LSuMgCC";
            sout << "2WRsJTQ6GU40AoOcUFDPOo/Gv99uWzDQaGgJJzIxPlKhT3EtJWH7ZiapmojgRgdS2iEAXb4adzRQ";
            sout << "/NsD80C6PvsuxB6NEravtrBV4nz7fXq2u1mPkjv+/jmp0hHf5SV1YDE+j+EfZqNMgEMaWJfMDbxs";
            sout << "C70Ji0d8iv7zyevr8fIrVmTbttOxS73NShBRX7mHQnUcDZFDfLJy464v8PEvwX0gr8/ytzdMJTrN";
            sout << "LEYHS/+ZxEN3CZyr5tzYL+DdqsTY4qFyuoQOQ0brplrGc2E1LubNhGVHXoKTsAE1+D9oq81HG4VH";
            sout << "Y91M0A3+ckyDvQ+dTtPx+KT0on8OkZzSLy0VUJ6yHVmzlzL2R2B/C8Gp2KHKk2YJLxE8DLPRkOyz";
            sout << "ZPrWyDUWKir87Cm6xHLScUj0MWujlg/ag1CLz38LKEBzoYxCzqQ/Zmyfp3NZRi+PEhSlSj8AyVPv";
            sout << "FOFacvTBn+dQwWq6uLbOBXJiycohjgBGxrLc7ILTp2FHmNfpELCTCowepYKuO8VnEBGlIUJAH7+I";
            sout << "zTtM2bvgr7k0b8dwh4SMdCBSOWPfYxuY8kVy/vcW0xKeAkrwKlVuL1+HJAtUTK0p9BOWl3bi7E4q";
            sout << "Ur5UH6CSaVzbn7QvGh1wSKbr48XdeOx/MlqvH6yKeXxT+iogx7BdmQAMOsndXWH+T+HBbmqIF6Z8";
            sout << "qA68Q6qcooNpHu13lqIojIbbf6VUT9eG9RqGPYXxKBAdWuRDbMmklXYoM7PI4VlmBcIgq+4U9Bps";
            sout << "Yge/Oqf6DPFxPaqG+FnYPl7GpcXHNHAsBh5JwaDAM5uKCq4Y720KHLKqG1buziPjRwUr/5LhXyu6";
            sout << "CF1ZzEKb6w26NDjD47myrFKRihB1zVZBxb6uIlHVcEqFZhJqC0uV/QQjRUfAQrG9G804Bcf7nuGM";
            sout << "6u+nXr1Xl+oQIRiEYueMD1T0WbiIYIIO6lT0ICkSQQjoLQXq/8KP21Yrutv3H4PjDZXOPi9Z6Rcl";
            sout << "40Vfdt9jyqeRgBhkhloRYFOBwXLYxlb6umBcmXoT3cv/Zlh4SfQrpG/LrfAHtYFyfwoDzXcV8Xuq";
            sout << "qAtTkj97UcAVnfC0lNrBnWCT2SieQUl8nNLoIQl9uhzRBFHH7U9ey6uX3zop4esABGV/DL1ypLd2";
            sout << "Yu2Q6XU+1j7wh3Rn5zttubqvYR8P5jMEqMUaX2Fith42jHy0U1NuNzX/MPCm6DFLvah/G/0sPKiP";
            sout << "lboucbKekedV/knGV3kx+h6MmMgBRC4OkkPf3HG5ewpVpe4I3SZ/auTk1ZnePM5RL5wskdGCdbND";
            sout << "njzEv5sKtTE1TxCAgVh6iFUtvr7c+3DNDkTilcm+ILRMwEr9vCF/OGcOT3YWr9cW/eP6fy/801bU";
            sout << "nhotDzcKSPQbbN7jd3WIe6kaeXjsBk6LIHmPjtGPT3UdRFhEJZrnABhQrasLPeY0/+dODC5Y3GcC";
            sout << "fV68kmensUcZ/5hFSEpLsScx/OtGvFI7hLNBAAJ6bhEuvgId7OXxQgWjQ6wKDhwOJmEB1Ra/Bztw";
            sout << "4bGdRMl6c99nGcc9GmSWI1CmQZqbNfxL7AKO1oSkwpWbK9E9Cl+ZV5XyP28AQky4KQlYBvwUDH57";
            sout << "LvtOvuIe/HzSagQjG8Sxf7wGF+MAz7MLRtFYGkHviL+mRvtp0NSHSYDEOqaIsnk1HbTWHUj0FUQt";
            sout << "AD14VkX9ReVi1LnJGJNwmjeLYk7keYTkksi/kKEWCkh/8Av2Rlk7oZCwODxsAcnAyXStJV0tEAHN";
            sout << "lq7MaAJz5zWZ4ZsETA5mn+3Qj4jIi5vbqJScxOh5KCvyOYfio9Eo+DPvyJtvyjGnqVkoFFk7v+8H";
            sout << "e4d20Aq0vfYGKPzVNoXAtOPuA6/d/2yizrl9gEPBGVw2I3Q1/XlFMIHHZ3yT1TGJfGXcVkrYRhDY";
            sout << "s8k6oEKbp0QQUiSkercwxbLpSMblFDaGeudPcWzKVwPE52rMEzh1/jr72s6ZUd/+Os2b3+CEg+lj";
            sout << "AlnkU1tNZVnanmzOb5xBnk1H+3Q+FppUe9MgOK73344O7QlbBklWclKizZIoRVaOeP7WvjXSJK/A";
            sout << "PPXXyKLMwP8eYDDPKOTA7MvBH3q4r7j7Au5av2sn1ZGwxBHLH4aYFH54kXJa5rwl0TgsTAHsQ2+5";
            sout << "0/OWixbr8Ysl2JxunMjcewhJFI3mLHFmsYfVYDwC2+hIubW231HCxvqI5BgWZZRU+9DZ2hdQ+orw";
            sout << "qocOI3yK5uHs6Qaxv9HPgWzNXLG4a1h2C+RbhIFMGNUcDAGrCgBCgepruLT0RLuaSYrsZY/gWRwj";
            sout << "krk1M69/lnwvZlasFruau26RLyMAbELixpeeGpztv/tJx5fqStsjMGLHUsRj0SKgXgdawRdgUXUs";
            sout << "Hepp9HsmJ/HR+5xAv8ORiEQekO02b/2wgqGTYXyHGT+RP9f76A90kVJ6Px4njrOLrICyTUcIFrIh";
            sout << "0AvbAEysPnxbtYcp5tdNbG2RuarygtgscOxERIa1NaunqnBouzW88mOLeTgBV9Ofh2fvX3qalfcs";
            sout << "xYY4//taBvyKWHiCanf5drYyOBPm1m8uzNP9ew6K6IYC6S4m8aklO+NrkoCqL2sq6Ged5y1nsWws";
            sout << "BLeSaCakjVDU3ysh+J8YamF1Tzp+cOfh9dAquG4sk2jJmSfgl31jtG2lCBkBRQf+Cybgr7QfJYGd";
            sout << "ZoX4RLzmSK3BXrufSNHntV7dXKCDT+2NvrW4n1EUegwonALERWRTMRDmA2NPZYgqr5cy8v0KQr0V";
            sout << "hvEMjqP/9SPJdTtbjJY7WtdTNU5Er4KdzasYcJjphAPU3zC5PtCHbtTVUGXNz1UneXnzrN13zlK4";
            sout << "WyJ9UcXeWpjtztobQZ/8NwXiYOndZ+/qF3BdYjHYDYhSjod4JyCnmLuy+cVG9yB3e21XVeUVC8sh";
            sout << "7yg2oBx+9yn1ZK5ResSEAmX+m6Jq0etJWVnYvnl6TSN0XAFbZRk5n6r9w443Frw8RVRfYyISbuTm";
            sout << "oLMaClZFf/XxxvCIQAd6+IDMzukUDvEKObp2+rbf7IWZtDUeR3VpCp3CJhTMr2UBRC68fwB3mx/n";
            sout << "C2pAyPNX8WbZ8ZpAbtW3ax8U+yh2rH+hEK6zJSXFZk0Ea9Yc4MNhtDqGEXLgBvPX/OMH4E5wJxTP";
            sout << "H070u7+OQWwA+Aeup//kJ+jm8EqF2RTAyGCiiYl8JXAMNBdIGaG+kzb9RzRD+YiyOrV/WyXn2pFv";
            sout << "68NBt9sunIBnzNMLCPzT3lb4DcoOExmoOjcrziN6G2HjRsrkmyVnjvgoWO2wILYkm+yGy+ZEM1U4";
            sout << "MpVYZgXikHOos/FR0GG9H65gUCPthPn448mtVNFvuYJhhenvuuSkUWI2IFW+rhEeOUXg+r4ZISpZ";
            sout << "F26+EzKiDYKJWbJQNrVJv7CygE6SZ6E4VnKuXbzKHDct3O5EHoE3NDCO7J5kD7RyFE6HgUXusSgb";
            sout << "kH2DOgJrK8KXhwcDIH4AmrPFukCmgS0/yTWOjiZKW8dmp9q9UHKQlRf1rqgmkph2fGXXBbi+AMLd";
            sout << "qtk8gsm5Gb/a/fHIg8zOJ295ZShCqWm87z9jK5lx0/kFZKraB2JleJ0ryPFXCOp60hUbSvzzfxJ7";
            sout << "JkH1dWoRvWr3wpXrwZucfHH11AeFRe3ShRaAKy42+CclzPkWDFLnJQt8NXSkZeuoPjcx9A7lI6UP";
            sout << "WfwYZowJkoGDUq//TSqzSK6XAvQBUHuviclx/4/C3BvCwu40wXDadUtt+2xWJjNNvMlhnoPY3/i0";
            sout << "yKS/BSg4TsdTceqfkt0nU5FeTdyfpLW/1TOAFERXkV2bTkEnoJ8g5LXHqj58hDZRnO45lxDXDYvX";
            sout << "/3G7ja5OfBA+8flE+MDTtRnWAxgUBDxlFEKD2T2RBgQlLcy0Xa23LgAD3qIWYH+o5UVAmVoI5HjL";
            sout << "mJ4FxQdvJOH7VwyIubukaioVoB63Ls1ySl1Ysk7EdgCpdiugM6I39QsFlYeHE3AlLeoHn4caWBFE";
            sout << "lWIhgKer7kBvPFG7M6bN5flAtzAbEXFAH4M35L/6Cl+O2h+BvCJFiK1W/LZw2pkc+Ie0xC0YNWSn";
            sout << "5nAPPxgGYtxCHLVl3R2XJvBcPpAHzbc05fAp2G4AKscWc75SZwfayKKYarX2U+zu3PVJIYbioSqD";
            sout << "B1b3Wq4vO7MDdD7AMZjJO1ZrX3+8piEVo1MeVbqZBy3VTbjsnkkErQfyOulxlQfdcblMj2Zn9NRg";
            sout << "EQwOnGEsWulIurqEe7bQEZldvTTz4pTOZ2W+uv8TqfnvtKgNESaqYvofF72ZFSJ4rMU+Z18Fqaep";
            sout << "5MitEMIKwX3ttFQlgM4/CWeaygkhyNMbFA2C/T3jIopsy3bPo7AwwhHxaZIo9ghkvt0nTP80dgtU";
            sout << "DhP2a2tg+fNUkvWFSfdP9/NKyH5Fahlm9UPNozDehlI0dzeHvqSpdYFYuTYtf1jggNVeC3r/Zfq8";
            sout << "+B7U2qq+e4iGW1KVkOlimC3tYTRfPj4ZILtHAUIwHh4HY7pTxxpErDLpxyEZJNkBUxgN0hPLm8UN";
            sout << "hTu3qFXzB149UpZCv/JmjmmEgLAWMT3PbBCebNVPEjfO6I59rTOvl+fVD3X2UJM1It4mh3NJPHzj";
            sout << "4O7qLNm+S+A00bZx/g4ncCxrCvFkTpBmt21ZMAI/H4swFDlJ0gzKTq6+nhI/nbiWsuRY8z+GNqcE";
            sout << "Snutk54hYXlQ3tVAxhRY4srAcEQMcHjX8pWrJUZjRiiYWnVfnmUf6U3rki8P25851GwTUGB6/lL+";
            sout << "6MWKQ+s1Sa71hc5icdJtxTJSWADg41KO8n1xQ0Pu0KJzvdNMM9G4AswY87XnASsaF9EXKDTSW4q/";
            sout << "s601Ghu8vEYKQVBNfJdiZ0wPsaLM2SvRwSS4Ji7agaJRLBvcv/cK0Hrxml20CNIGS2Q81xGS0Hdk";
            sout << "ykMptYe+8pClFugfpJ+ETSqFgclYc0XoucOUxFh0vyg5CVE7WX6chpBmdhwWNnoyJz+WvNcoPIu9";
            sout << "UZdYaseFLVhArb8cVaA9Mfk7tmFxaxNeXsIBFjiE4dcWKJbZXMnkhCaqG4k3SIHsswxjtj9hFoTB";
            sout << "BzmbIwFhxPg7sZVrG7Af26CYC01P2PGqNqJnZRZQt1LwtPVQHzvMk2v0r1I3DDD6ugGlva+PrHIP";
            sout << "D8FCY/mc0poDzTANlwAx28hkPTbtCpJIrWScZ/Vu/KYJo3F1Gk4C4CwcghgkwTYLhe2eMwlnA+Ww";
            sout << "vOlw5SD9jca7GrTA2tkTvsUnlskDgGlkAvEwc6N5DkS6clO1XRbh1mwhr4UwRhkR9Z8sQLLUv0yt";
            sout << "N/Wq4i6xuXuROdy78DSiSh9Gis5XQbqCvf1VKUOKkaA+/H0Y+XsrHrRCcqE5uaY0iIZtc62XgVHW";
            sout << "QyrlDHsh4Lt9qGD93Dx2EZQqyl8KoyhQ/WbgnG/s77zdSNGTkoJDEQKIrKLRk9ReptdqQzjLPJvf";
            sout << "GwN7wgO2N68B9XbX8hfXUmHX+G7kVnncugQzg0DS1qQ0Hbp4ibZHKEAUHtkoPEVgzGcsVowGYCqB";
            sout << "KCGaq6FqyyIYp+UO5j00xyftvh6uta3atuu+JHkWAPGKY7uooV3MGmdcnnF8umE1NDEBfrcWrrNi";
            sout << "IzQcnZe31Nxqo3WsknqYpRvCDwUZiU1f/EidxhykoQ1NCo1CY6ociQna6kpsM52E9ALvYlUyobN+";
            sout << "7iJQcfagbJw6OpT+P54HFsedkIAlBgTIfZfag1u4lKZnibEZ5SEBfOGeI4pemf/ST+4a+2GalVHc";
            sout << "4W5T3xhuJjQiIeE8X2/nEwqEPQRHdTH109EW4BCv1rlh+XWqRXQVHPOYY3lVV4ONn7/iWN0fslOG";
            sout << "z2/MghTZb7Z1MhiWLxTcwpUXoyy/fexdBySIUf2ukM45ZPLr7T1aCrc+pW+XKfTewxqhWnKSjpVR";
            sout << "o4doFMv+eVk8mQPFjAP98MdIdYEgTakSQPoEdHijyq31ID0lI3qDPElGdFEq/yKpd9Tg0i2OaOvk";
            sout << "BiaBfldrXL+7jlCSTB14Vo0RCiGIGuxqmEgQhUrT2iHV1baKVWUuKERauwS3jVNz+xq6PsAU9lWn";
            sout << "mC1WbmHOWyHpvoxE/X3X53njIVVIbRnGwEma/1THUaFIZ+qb4UVfVqBXHCiCtyAnf7+aH2/T52Pd";
            sout << "gayp9h72ofbN/eBLbc0qXqECKcWLvwQlgiUkV5490CkjKnf49ubdFKRRH0PKCi0CR8T8lfhrHYnf";
            sout << "j29CDEgwX3LkIPw9GXt93ua7XgOilyqxhPm5vDXFpfgR98Y4iMUyheHms8U81xbRmM7SqAPXDJsO";
            sout << "vjoNtHmc20hgCb46BUQbMQMe/hhDjQNFMQ+HA83cvTyCqsXVdKcWIw3vPfD3RUqZ1R6LGgmLe+Xl";
            sout << "vS2S/uDJjBohq5pJ7xCSI2+ylMf47qtS1pYd1bUXssnpJOu3kplTwqW5idkz8l01FlR+duXcn4we";
            sout << "Wop+0ncMd5AsBpk2j868TAPVEJyC2lyQ9Qm6OouPEtMg4x0jyANl7JhyfEesTTvxIm3GNFe0fKTT";
            sout << "cFwOojUje215zqgOqwE0b7JkipU4ssQ6j1lSHuW8lWEVKa17Wr2B77lj21JxubQJ99iifjoarWGs";
            sout << "5NX5nxCQKJGDEPagDVmL7TZaQj+Cp34MR9mCJtRTyLmA4g2rwARMsbq1btaKXUG91484ipcu2jot";
            sout << "/F3wLGdAhXQFWrxhIJS0ORxalPcSZ01zkECazBTIMokBrneVaX0RulGEMrs+CIGiSE6pFJKX18xs";
            sout << "6dGqxF6TkPk3GalrQNn0al/8FgetsD8zUX+d5PdqOpb7qNqEfHAClOcNxhv341nNrGnR7YuO+r8R";
            sout << "67gAjqnan2NY02JmVhT7xqN1Uz9jCWqmbLNAPDUWPvVLvFtHAqqp556xhpVhJchWi81mIZJ3S2L2";
            sout << "TKd+jGHjojiYNyjQxz3DD1S6ZN4mPgzK6JMC4txgvcQ9qoWhoqBDRDrj9s3lT5kkjZnHRPO0RE+s";
            sout << "W6m51Z6sKMT1/yoneuJEnrwLxlD3Efd/cq8zmmUKREZgqR0q7WkXC1Tfs5sPVwHDlc45p2ejbze2";
            sout << "iDEv4ULiYrNnj/wdOSwVnNt/1DXvQ1FLtJ7K/Bt0qUszCNnNu6ZOwhOM2bqVugcOvONb6fLNOzY3";
            sout << "WYA+2RGS0nuHC/pFuzX/AQiRveFsq3IqtYxyexv8uxmlg4sPDb7Q8i3pT5fQDYAh6CIFX7mCITvy";
            sout << "2JnjSfiVqGdITtvsHyJtUVO0YlQKUETRBzsUQ7bvQ9gTETskkHNPe3h/VR/Dpa6ukhyspf1G9XI5";
            sout << "bobOkgu1/3ExBOTiCjbcxgGrgXw7VdmFURpq+FAQwuNxLDSoNfwFw6ISgP80lgBDV8/5l24p517f";
            sout << "fvNPTCus2I2A//vLxMIQqU6cz+kIeViDq0fcTxoBiD7SoISwJiRTqciiJ005uVYFAisBpU06k0NJ";
            sout << "NQf1DfYz5JKAiP05ehMhQLCnJaHjbC3yILIfxXYu4wEV2lfLWWZ2u7/0oDBaZKNHv5JLzjQglqBr";
            sout << "o+0GHf/hhu1EEqAyFRPWruxhJ1XzvbLt8sHc6wsxkQCYXPGfxlz5+WMrSdNP6jPoEleH7xcL/b1r";
            sout << "mw5oP8fsoppeoxiK63Td0Ut6WtCI9grmCKOYJt9UfzTYI4TLDsI7mtofZPUvX8IeOOr8LySclV+m";
            sout << "AcwU6BJepTxRmINnck9tCe7m5unJI8nBy+uy6a9NvSILGvuoJ6bidAqS0dojvAV9m1smC//ZJPLJ";
            sout << "8UMVkhMUHEgx8n/Ss08rQXBFFqao1rCOvbUR1XC6g31GAju2dLm8Zyk9eomFmdOdtRTobmK4XbX4";
            sout << "mNwCOlbKSZt1aBDN8JLYBR84cEBB4sjsPEXupIrJyKAe306c3RrnYj5lMxvnnknMqIfllkKr1BOm";
            sout << "kBYJ91aDe10mUd7zHQwW8KAjHI4pmDhLwusleQcl0RL2Q6CgOB3x6ZaYnXLtuWsOf958+niS6dg1";
            sout << "RA2Lv57ajSAOHzvuy7h/WV6uWhfsikjw6TEci8rQ6v5gY2DEMjrtSCbJeiJzcIqIC8bz4lvmEfiV";
            sout << "QpZPhGDgfS73qeZV7ljrfBcjvSuN9MPbMFQfkr5v9lTNJ+/AosXAjqM6aJ4TTrMq3XAYMcbuEaDt";
            sout << "89cI2TabaBe8B7cniD+JBOu73fndg6YglAy0Cl41GFjU0u/xq0xHIim7e9TLVtTJE47CjTWRrBEX";
            sout << "ZAJLPDlnhevjsz+0vEuLeqGnsX5yjbmtZzpkDvh+R6eZAJiBVq9uOLHbKqplwbjU31y2OO+Gf2Sg";
            sout << "C8nczxwSt6JT8ktG3CuGhuEIGi4l5LAjIjYd4LpQVcsUM/vP5cbzaC7/XyLmSY92KfBD8OuUL6FJ";
            sout << "kMNyHHEtawNcKlUWsW6N7ybRpvZEwRXhQP+5Q3QXHDbiQ9YJhvnnLWHFJx1TmMlhAQlyJBnuGD9q";
            sout << "Re5kqVi5ztXFHLY+yHAFHgCVbGq7WS3xxwQD+jeuLEnvbtNF5qUePj7u7psaPpYkQEj2DvRBoBGm";
            sout << "eEheol5Gc8NcW8wRw0hLr3Syw/8b+1uMl7c/Sxx08yvnB7e8JNQ4kQMkxVWFCWE+OXKul5nI5mBN";
            sout << "HRbSO5bBCuYU/9P7DvvK2U/WmcnOvNuFFax6mmBc7E8nqao9wKmpypDvddWJ+aStllc1xZt875G+";
            sout << "xrYOX7Iv0MSfk41j+VwrawGyVYKLrssBQoft2lbeDqOsJBgxS1wFmxQXgFl5pW2ynnye7GuR9A+t";
            sout << "pbgKYqFhYg8v2ZIYJ/SDJey5H6TImYlPNriXZv/ocn888rvlfOBjDcG9KiJ/CJDLs6RdMMI67EOE";
            sout << "5VBRISu61OMlhv/zp2bOsyvONztlHXERM+N6zViIlmbewanZ7ujo969A8Edx1YiD7eBR9AkB2VxW";
            sout << "nlnX2RzojsFbKIQjuE5NoFZdE49PsauvRhzYBid47XuPLu48IGlVWuX8dUENwWe8d3jUb3c8BPUv";
            sout << "2E2TUEOcbEx/b+q5ex28LRk3CL2AolCUHMhw/aFKOj+P+AJMC2qAgzhEkVUWLfD5oHnN0gFzIr28";
            sout << "EONuinHzkLIW/FcvxqVjNBn9O6svsz1RGQ7dOUStEjwpKDyYWboawNC12lNgh6+JO5uA9jGqmSyF";
            sout << "uBacrq7GoETHISP5/7Mc6PkdsT3IQ7iy+suT5vTybsqlATrvJCR2/6S+M8kZCJdkIN3tVrbKqKHf";
            sout << "O5sHBb6MgPVWotsXQyZ5m5t5x2bquGfonmlEdOcjBpYJSEM9Gd+8zvt36NgjRRr7Zgs8DU691FOE";
            sout << "eoOOYDfc5yzjHClH/b2k5po2saebjyh6Zf2/EC5mQuqkHEeLtRbxSDSa0u5jbh80vEZ//J/3nhsr";
            sout << "lL7Am8nXsj6aqyEFVYnomyDo9Uk+KLWD11PqLqJiCNTnxZbsCwiqZDo6qo4JngJbvdbGZBQMpgzU";
            sout << "kj5cDYxSZLDARNLr+/M49l7esBKTtJUVtgJX2+nejrlgeNAc2BwGQPecI9Zx9v7qHOhxSsCwSrGG";
            sout << "ZrHrTT3Pb08AKSzLQLmuijTWBRw9SmApDxTcUnSCI3RGTmqueOtpq8hGzD07OFhFU62SAdx8cW8w";
            sout << "oB/FAeZGOpIHY3hi47JXjZGTw3oGMtUkbq6RN6wQ5NJzij/HHjLzXyXKpd6CveDuxM24zzyvhEwY";
            sout << "Q3FNWhhgi8RcIFb5VutGDNZ4cd+jqf/veLh9Q1110CuDpR7MUUWm+MlG//rMcScEZDOZORL/qMuw";
            sout << "+aKT4+x2LXKb8G5uWwNovb3eZOJDTnb7Mvp4RfFql4X38BUjUgjZEAO5QXRPv+OfUi3PeeZBDQO9";
            sout << "yjqeCNsecLNlu4cf222h7Et1rdp8zYPwMbdfLrI3pH69VxM5eSJ81OtprdbsO36svGkPOOB8/rNq";
            sout << "Lu/xFCLKvWnHCPoGT5ng1vmcUl6Lkz3r+yD3+Uy0X96++sXHnOVdiWxKyRhew+3/gSQuM6zoPumc";
            sout << "xzAkEjOiKghShOmqPKQzrsEUi+sZ3NnCBwRyUwlaFw7/Fp1pi/N/9bCX30AEfaD7ucsxJGtC78W1";
            sout << "jWxi5K4qPpgaM+qU9VORlnJRFpYFBgu/nqvd8LkFrLkLsFDxkEO7bQDjEKX6xTltM7nbF/KVSasE";
            sout << "OXiMk4x9aOa8buelZrRiANp56cidcnJ8Ayv1GGAaA+hNYC+BNmTXZUal3NyMvxJOzFlKfaSUbgNB";
            sout << "EsYQusXQuVqkD6lW2Odx1Lt3hHzF0MYgMY3cuw5oeMKzarQoA97JDf74Lnp63MeuU+pjBanY5pNF";
            sout << "9D3/l1UIsrFh2ZyC0vS4i+5SGDI0Fza9ZsopX14fbUOG5FFSkrHPL1ArMvIXAG7B3OPSRF9ChX1u";
            sout << "uPCaRK+0JYxLdACdJB4x+Mhnv3fS8w5dDetUO2zy6swM2OonEXqln7lETWsBk5yQpXBo9sR8RamA";
            sout << "nt4HQhwaK0IAX4L6p2LMKJ27YUq2F3Xe2PWf9gMXdrUQPTvpq+MdEnDIJYf7sTkT+lyqLJJZap2y";
            sout << "jmuv5U6uQyaeJ2ayTd2TcvryYtrYg4dUEfTvjdlE1QdJPxUPCzjbkM/y9/SrfPCXsWVlDZDtytup";
            sout << "tSK5GBlCH8JOpiiwgyBR5nAHNZ8SKP/e0K2Ps+mDx6xcSz4WL6loJUN5+lPuMHYxXbT53LPOqwNX";
            sout << "nS+OCDPFclYZO/0TxEerFNvxvjXLU9VxB87zLZRV24VAztABk3d5zrY48iSZ8WDUudPF0mf39geE";
            sout << "4j9zhG2/3+rc1yYFhAQFN0ch0G+Gu/mEEkSkxSZy/PK/v+ITne9JIqao8JfEVgwmQY2VCSyy4Rj/";
            sout << "KHb5SOY6CDIWIRMRj7GsqxI=";

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

        // This function returns the contents of the file 'frame_000101.bmp'
        static const std::string get_decoded_string_frame_000101()
        {
            dlib::base64 base64_coder;
            dlib::compress_stream::kernel_1ea compressor;
            std::ostringstream sout;
            std::istringstream sin;

            // The base64 encoded data from the file '..\..\examples\video_frames\frame_000101.bmp' we want to decode and return.
            sout << "Qld+lmlZhXVX5NDRWIFG1T4+OGdJbkmxAXXdHZEtpDf6knVTlRWyAhgv85Tf11KJZbhKv1dcsKtQ";
            sout << "fX3y/9RxoGs2WfSxuQ88PN3jZemvW7PfY1s1vwY1Govp+hD6LGyg6qbJF5iSr2edUHLWGq/00D7J";
            sout << "dhKHu/QsLpEM3w0yjnufc6hORX4GNjJ2a/k5xFnHf29G4ztGVp6OAlsQa4tcLsh3cOK/eyCjevG6";
            sout << "7GaMl/sfkK4TsjNfqdXM8sWxBCw/r6JTLrvlzVxDTmu6Sm2MpQ+IrRXFn4gK5vwPmc/mEp6etj4w";
            sout << "YeFI6w2jGVTFIg3R3ixaPUtvv+gaIOtPsF+4StbI3AlmD1kDBt0t2xEyPOsmjOBuh+oy/si3xWt7";
            sout << "RlCBakbDZnIihbzDMrLwTJvgoiQDgT3U4JRpH+5cL8E/3M5JAOuKWJU9nfdtNVIf1NuHCd1Ajsvt";
            sout << "LjRLI3+rqQErHsKre5sId70rx+sYSqTWaiiGi0Nwg+JdCRuA1OwFmK4rahCAEjFr55py0iZxFuo/";
            sout << "r9DLOVGqKKIWDX2HQQrQ8N1gATa5Wh9ypd1RWKz0s1wghRS8nzIpFqtvXxJxYz39gjai7qUOP25M";
            sout << "zEL4RlnFsqP7GJaT7YnpV4PLkKAzYoq2SMTq4nlK2MGohKoPNj1JCfVcq3syGMJNYtNNOtN1m/E2";
            sout << "3AKUnpPX0CAEgA43Y5yMLR5ca4+KGlvO/qWwex/bcU6uZSc6uEh2pcVaFvlnyUYdZlBg5huymL56";
            sout << "HojkUQWiv48vxAoYtiEc2xsgsD/HeJxF7z+kWJ5nDeyxLhCLpDZfrc0g76TVEOVy68CtdGLuhDe0";
            sout << "dhFhj+tlkmbNse+X7PlhhnI3Oq8kt8tFJly0YH9RXYd2eQDSK3Wz88jGhe9wcBEwcFir1foxdRa/";
            sout << "rj6qz9DeAo6bVEZkIEXPSXL9EmiyTxwqjHgIySK1il+csxW2Gpwb1hwlKQWr2x7bRf826W/DGplc";
            sout << "XNLAy0HLzQh+8E1iJZhpCwYthe751RQLrYaBxlFNZtlHDKIfuGg6ByiFdr8y3T9dx1e4xJHHAEhE";
            sout << "Aem48B79Zp5xD/ZZOgAmJx/hrXJCnYXdH1/5FNhTyySNQKG2fChxHwALxDoOsB5BwFP77RkEuXkO";
            sout << "QWLOWFUtu8XYAvf2re0ZafPDonS/GoPZkMBWo4S6KNoczfDskSXQ4PqKAW2stt1K3BoxO5LXfQYG";
            sout << "qepdQBcZfL1Ir2WmA0L2HC3UmFr4bPwVkzaYCs4uQJqivpw1lD7CcNKi4BfBqWFTs0uWXQrwY1Yv";
            sout << "rmhEavMquYjJRdo2QnufXkSjrW8VpVEthiQK895ABfggCWg4F3yiqBlCR+aGcCetNoJZ9pS3sb3Z";
            sout << "32tmtN4rrQ4jMb02QNnXPQ+vAavQaWok/dELaGYCG7PefjjcFqX48WNVmHQx+wLW4m0xo0mqXSxz";
            sout << "SQakpCGETfxuzyIlmYVRh41QcDdd/NH/CIwIMFVyga44Ewpcuew5mbz7bXP3k+FlxzXJ4e0ABVn9";
            sout << "hO2sOG1V8teDymKMNPQOBuec5AenB+kchCtCDL8ZtglpBBqmijUCegyxiVpjnCzdGQvIq1XdfVmg";
            sout << "e/hWiXT1Vi0+1wtdL3CayA1h+zvUiIJspWD5QOHj8O6buKKWky+V2UZDPK24AuIl125ajH481E6H";
            sout << "sjS0/FqKjHWfw/SayHcCdbqIt6/8AouwlkzGOK4C14HxAg1jmvGnRn+46SCXNjlHBDH8/mJ0Z1uw";
            sout << "C2Fv6G6t/IX1hcjJ8NBQOY/3gXAtDRmkNUc3KImzaKNDUkS9N8U5QLFVqrBQ9Dw2kjJfc28QJcXy";
            sout << "OOItYP1RQmG942DKVwW+QQV/fvK8BFpRP5yuNgmzjGZI8gP5cqIOkW+D4eBFBb8PYsa15fOF9AWB";
            sout << "RXDztTbO3uXWcTpa5w/dSbawwBqfeOY5wh3lnYd16Q+Yorn9D0O9VGFWr2JqKwl/NcRtH5FFsNC4";
            sout << "LpqcX12SoO3gsql2isJzJlyTmeSDB5C7o0mBE23CsBx9uqDwBJn+YAucIFxB6I4sxgF6EOilkA0B";
            sout << "5CpLkspGnHs66pvrLeByoDMCpkt56bXRlcvFhjRo5nljY+GjBh6Ux3ZJnMdUG4Qded7oVaY65JVX";
            sout << "MpoHUUG/VSR8FWuHUP+SXh71QiEjsrAAVQ+Fm+fAH3xQ9Vjiv1ffRvxkb72G9O/LjF6NWWgowFcj";
            sout << "QBhuD/mTI2q+6spFLWlZk9PmCWTBkQGsWouyZMuVQqr56sP0UEKANEGwpq+h3pq9t4+eN8WcGhyd";
            sout << "kKgOgUvt8aY7Ib1HdckQt9cbtOrYyjxrQ7o7CXR8eg1SLPDJoWc05YevckkECeF4nQcZvoRxOlEV";
            sout << "rQwpnngQ9oNoREpNnGlZGtBr97bvWTw3UWwiEkqDJWTTbgoDSzjEroYV3RjVyVKu7BZRTvWDFz+/";
            sout << "xohrwbZ/NFA8wGZpgm0qtJRMCyDUIAJuQSEjxa925hWfdkVJxhExxXgWp7NQzhawF7/16Fa1tyMK";
            sout << "62nnOYWV/OWcNOrRCt96RlQooSf0DbUqtLt0f6AzkjKobn3qpkaMxg4Fwzru+BaLmId2Cnsqdlvu";
            sout << "jCjsgpY9cfv7NdcnPvhLSPyykMPmBRe2XZycAzmQdpY04DpVI/A7gpkVyH3IvYP0ZC4SXNwt3Iht";
            sout << "CnmDh+VF9+iVjlaG/JJ2cSXG1mDSfjPgQqEO2DtotkjN3cHmqqSkLqbmA2PMnbzCn5FJbdhES/C7";
            sout << "uMRFLhsEyH7BF/UGHCyIkzbGQhF1q3PE1PP7A6UOtWr9tmKvctlDnsiEpGvxBN/8wfDERqImT7Ex";
            sout << "f/0fF7UbHD1+lJc9VzXKeV2ZsFhFRp7r89yVcejSwiAcgHenFuvMEtZ3lgRqy7mFxwN7KNAh3O6Y";
            sout << "RqGFSz9w0gb2gmX/1f7kVvoKTgibMHYXyHUSvusrjX9betzQ4NysjILBta51/ZbeBIZPF8V7mhNE";
            sout << "5c6iBo4j1i+jEzBGCbH4jGwNTRoD2S+uWFrDS2x3hYJL6uBoEWHN0oG5f7H3I82zkju7jkuUCM7V";
            sout << "bd02Suu/hjwDlr4YaQUTIo2aThnXT7ZgCaNhkWODTVU3BqX6M8WP3zFZrMWGtbuXrE6WdweK5ENk";
            sout << "+aV+0CDcPHertb4LmJCOY3yXtPe15OsDHwedxxBJCt4y1UdamFtFo57XjHeAr5c17B/MWQ82/ZlW";
            sout << "1Rwrm+gNysxVGgfmzvmT0hpIiE//g/HqI737ovjOo6yRfkjil8AqIlOT8Gw8mBbnME2OyjX/TPOw";
            sout << "wX4F52tAhs3oi9HmaKHW//GlC7ILPqFN1Yg1azmvTgqVeYq3l76JbgZwNmPUEM5z/EfOcWu1GKT1";
            sout << "Pc0Hz6WcPYYyANIlmv32CkBiZQwMi5AIhrUJvpGaD16hPW8f5lXO+oG3FSJlhZhiIQVKFX/8UqZk";
            sout << "AzFhmyPjUHNHOsJo+1c+5FTnPsjHDzYwW5fUQRSz3L3/rv6AY7g0BXKLmxCJYzIwxyQ5i7z810d0";
            sout << "jTPy1DANcssoEqdA0P2u6udO3d0rtL1AY7dKoH5QYyKrOjx+HkLZcfxX2lpjp+kbq4uy89JYfo25";
            sout << "M4tX0B45FKrUDDtIPKR9GRfb6VgsH2bi7aediwVXCoaTxPNgUyKaFWSf+y9gPBiD060TccAZfCqQ";
            sout << "HqI2fQUwEfsiv1XeTLGH8mNIHETMKP78LdWLhATM3ejl8aGcNxsRAbMXm9JgJmuHxaNViUdWmfPU";
            sout << "hJMksO1JBS66hcn0jBJhZEYF8qUwG855G65k8Vvo8tihDIbJhxGMDByP1beBLs6uoohknWTtDU0K";
            sout << "Q70nziEj+VDbaexOU+dF/b/ywH7Q+KUAFvHH/xObyc0/maoRS/e0pfyhmB4OeGsbWAs9aGQNIaAn";
            sout << "Toh1zF+xQBFMKGurc3P4aPvh3Vo/cw2hQTMYJpUJQvhYx8XAnA+hRtnXZtHD2Jas88G0JXCCX5EG";
            sout << "9te/j6MEP9Xwb/TjVLDGOc4IeMYqVftVFcmL9hrX6nk3TYDyUkvLjgn8vXEmJ7qB8CthBF7UUyjU";
            sout << "mcz3agjf1/6xrIotky+zbEAy45LvmZkcTfVVJt5nVyir5hvcGyXWUOpqp/9NQr0ClEVhzfD2d1cd";
            sout << "xSSLaeTKsr0jTWFV9GDeIhprg4lJhxL/HbxFu3iOocrysDTTxsbLSTjO33ndrGiz08uW96K1QY2F";
            sout << "lqmK8FOhR8C1eaUdUFfIa/cINoThxQRkoYq3tUr313+VzGTRO0I2SSJMIHgFBslLDHwX5AbkpJPs";
            sout << "7jCQESWTp+LDoN6g2X+RmiLlPQiU7iJXxF7Y2SU09kmHOg7HFPKWZQ8bI+m6CG7kvBF9lWNuzfr0";
            sout << "lWz08lEMExIEAAngM7G0LqqOuLJO+Dpk+lLWjO5OICoxQ/M7akvSJHtc2mVOhXNskHOYG7ZtEEEd";
            sout << "ceDN4bzzJ6qEuYMljZcKkh3NduZbETGnW7D7Ec/UqU6WNX01iMGt+4lCWtu8NHySWGqIXcX+8ITJ";
            sout << "euDFM4B86RSKk38VpVUXXLWd9GPUYa795WcoVxHlFRPULnHJK0G2/AUuKe/K3CxHNqnxsk1aGdxS";
            sout << "NGVyQfhXJBWBA22f0Eclu/Z/UdzpKZjCZFFZWt/4ppIWGyMvxheOSqjA55keu3QoFj+xhE0TUNlv";
            sout << "cYMT/b9kHJnHQ+j9X7ReicHMNdtWYmzybolMFV5P8fBwlReR6BS7nAk7hu37K4fpKoKujfMUfE76";
            sout << "PYTfRoTxKKY0atJg87ZhID745jym89xIqFqex1vqb3Ysw0+dZDWHbwiidUZjHza33U07hl66qo7M";
            sout << "iHWjOPzrMEhad9fsvLGQkA3WNjj4fjx5TLN0mfieIPgAjrQvZhCO9DspEWW+jwRP63BYUU3rB2sV";
            sout << "oEJfsyRSoam9ujGE2LF6ePYZhOOs41OHGsbUweduJ47XGdt8Z8+wxnZ0ykwvxc4eVsNbNERVJ0pz";
            sout << "5hULsfB5BCs0jc2cz6N+1MLS397qKGNwnim8OBuU90wU0vMy+QWF78OnVb9jlg3asOK5riTvCWMb";
            sout << "qgj1qPo0GfWRojb6L2JaKiQlU7r5LiJSmmCQLdy7qnbtc2ul9kKn7iQZqxZVlHJklsrsOCdA3/Lb";
            sout << "kzpWUbCp21oyQKgp8PRkMv4NhWt8JkushMklTjuvDhvqVTqrzFQE/UJoxWEBEjeZWRgTG9gYISJz";
            sout << "9mnT+89BXF+oSahu0YO8TKgNMqIDQ0d5GVrc/lakm/jQpZ3nl2tVbN5vLMtCuwbkeYRiFuaQKt8n";
            sout << "+0HpFoECCJZdk6vyVFpIHNPSP7bfnsrZJHRiprvhz41LowOajR20mGW/+mlqo5K5eVxrW4I2Uumc";
            sout << "Xxp+DK5yhhIdYxY1SkRYi4CynSNUPLqoD3RaTFKfo+aGwd+N1abtMJpWmE/Xp5k3NNHXWi95ltoF";
            sout << "tU4BxGQWWDjG3UB9t6eUDoV/WXwCy5pxs4rbXLb2O9CM2HaBC+lDaW9RjxpOjI1jvAXmcQrs/MeT";
            sout << "ex8n5RxFcyRzjbaWhd/V4vZ5+qY6eLoT/cUpYVOp0w3lQEBaGz0W8bcetjroDYGSV1U+3nBgcKPb";
            sout << "rW1wp9l+x/ihlfM4yHdApD1WUTYfdIgnO6YAw3tlm0UxARb3WxyVjIQ7JMr1Xgle4UbFSWrjI8pD";
            sout << "U3arZPJzf7oN1eFDdS/V4fZ7I/B+j580QOmGI5LGvSBVR3Ic2iVSbZuk2mjWgK/YW4vg3kp7LjZe";
            sout << "yIxulHxWGysi//Oe7bqcJsM8Jndw3sId9UYV9r9GaBGBf+E2YWibCRJCiDjzO4PizCVOnTGJlOMI";
            sout << "Y7BvFi0d0eLHGAfIm+3d1bUqD1XcPblx0IYVa5AHY90JfMS7gBJHmJiY9Gv9+WtQmLZmNkU6Tcy6";
            sout << "yNyIMaIfI3UO03ApmhM9babcNpX8xhxYotZXzuxf19nnSdyOebzl40qU4BPzwwQmQMISK35A80xS";
            sout << "dvnVDMbcRQIC4fjwo/OKNWtk2oLY9ERJ6ixMMxG532tF12jN4ZbWn1X1HGFNXvSU2bFhuOLMSgDC";
            sout << "1poVi4Cjjgd4n49BJmlkeBoL88z83sPikchLnW1kxHubG/0VBwyHYBmocPrK46PL1rx1+dSwc+Da";
            sout << "3E+edMWBVe/ca2Z2lBDFIvo29fjwwumf5Ljg/M2YybU9jZC4vLR9VcdPJ9J/gi53iNOktmDcBcp8";
            sout << "ThCOSXuaDyKBtQ6nvd3AUGdNdLADPuNXTRNybM+Lqc4k7cApgM6DZV3Elpz38473d3HDAW4pCOD3";
            sout << "K/y2pYyog+/27OSOW9SGoVkOuqOBKwqrFAuD1jIbI/yDq/LYajcJIPqhkUv3srTxHkPiOBbFy4pI";
            sout << "6NAftTHf4GGy8VTGIzeD+z7L1qJToCogS+FoBG8ixRGUYKvUYEQyc1EhXdoPOYY6pZiMsA4xSGyN";
            sout << "aU9RrfweZH3ld9QU9Y2kjSqVOhQavzIPtD2wQBIDWxk3/bmH1m76qrcEfY9WKCb7Sl//1oILVf/N";
            sout << "b0/TZEqVSAFOMoTzTXO1ClXymBTA9b1bJKlQL/8DRayUN0NUMllrwHT1PGOmpoJ+AyhUOjEmWREv";
            sout << "mKkaxiRkpyLdgKyphJhtYGid41FAAKacNN4CMl9W+fZnydgR0SDvYvpOwveSXr66xfDZlQti8ZR5";
            sout << "i8Il6sq3+2ybJj/oakohjWPxMAA2rEvsOYkuUbviTAYQOM3jMXAVPIkgAYgwwXhvTpHjeRurdxho";
            sout << "5dz5zGTwSRsxCUn6QSTtt4DhroBs4xBCmYa8BFsUAzG9nG+SP+ejIgc6HsrLCxZ9Orlgn5nkX26K";
            sout << "bI2dWHqUL8DIgM9OjrXeWath283tUxYQlnQ8XJ2/IsmRkbq7eNJfBATvOmvk6CTJJdGrK3iX/xaI";
            sout << "WTJ8J4qY19ehkQoD8fLA0W9FitmfQyLGzxFSI4b3YOcl5KQUvSDjPpG053Ok4LhcQCILRhoHk5jl";
            sout << "usMw7nWyw97HV8XHrQkjsGyvkT0G0LyeNV5iYo6o8yFtF/s44/mfI3sh+b5AzhdYOEyRcnPSlr7y";
            sout << "axPlZnAfZRfqccQnEricrTtaPvYaJIh/dbtnlM5crcub0tJ5WNWAlLdS6IShjiIivEzqakfqg1GD";
            sout << "w4TzFl4VeqaTR7JWeGT+yyRwTfqkIEFWp4/7DyRSXF1hbiUifVaBwy4mpar2tHuSL/eknEI2Hd9A";
            sout << "oIFg0F1y0Jg4uAewz1MvmFsfoHpVAT8ejEF5OdFjkeUL8fjSBlX2SwMNaV07UlWkDDGckuHch6r0";
            sout << "FIDtx71pldsL2ALaQgKk+85QC8YqJdzgtfK/lRT0MDYOh4NlCZoFqQAImSYqasRtsyeNYWVi7Hh5";
            sout << "HUcaH4dpUrWFu8HFXN20rzWwkplYAGTeG1/S6T/sBXzAcv78VXwd8ec3+Soa/ZZNXY+yWuHziyrP";
            sout << "K7FFd/fcYHML+2bB+E2Trm4MVpzv7n0a+Kuh9sy0SVe0IuVJMo68R2Ftl7tGgF+dTKVKmEHLHqcR";
            sout << "fSjxUAto2wxtM3Lormd/1Yz3loH8CL83JrK1f699TQTZee6voQPJvlvKJlctr3NeWB0uwk3TIACf";
            sout << "198pytEpHGhlLByCzpwV68aPjs9EV5yekrCJzEp0arZSIzfI/cgwYAZ35Ukff/bp6jxg6AB6yUL8";
            sout << "Muq0rqGDqwpfTfTDaeBJAhwevEMyapDwrbzwHhnJLi7ZT5l67Q6Xjo/A8/U60t9m2Sdm5ULwzBdo";
            sout << "KdSBd9S1UUueBxrx109Yh1gjvbk6k6d8L/Tuqjue77ZCUV5mOJ/rmuTi6OEDIKdzZuBgooisaIg+";
            sout << "CargeAu0U/JS54e+b+Emr/56cogvgeJo2QeIKT5yhqHxtoEvHjhA+7q8MGuosAeQtp/9yYt5kPiM";
            sout << "j0d5GjKTviHQqgISzuskcAWAhjIfXSyVrFhiL2tZ9hqS0u3juXdBoUy3nXcx9WF1tLkyxONcILs9";
            sout << "+dOGRg+VpsTcjRqQyJoKm6OzVoN5J8iWKFkdZGLm0IM+p7F+jRasFIgJd4iPjHMsxYBlFX/aNcxT";
            sout << "dt1W97L7uw42LMYVG+w3wnHdkO/ddrsp8DemUQzsT7yGDOhg5LMBqK+Lh1tcizQrb73NDpqjGmeh";
            sout << "wmSb9GCzxDxw9uxfaZBBbO7M4KXJiANLFF86djhSpYgmApyhO1QcTSl2UR4XptqikYbPGFFQzKIC";
            sout << "qjvmsXWnECWWSZK6pnNnHWZOMto8VUmPx64HIPe1hW7KeRM4ra7J5Imw6F5APG4+Jg8IUf/sH86g";
            sout << "v3F35nnPnYeVvkQuYP3iLNapQqaKR+pQizKxXj/wgPagwSRrzlSDbKM7KqwbvdBGVMGfeW4PTpuP";
            sout << "FxtmMKzWMWAGUtPSBHktXyt2yn/Gp7HVfirM1FYaIOsWvNPtBfsQ7HGA3dvKbP9f6ZvRSonW/+kU";
            sout << "S+jp/UP+sIfihswVxrP8TO0YqQCwz6+X3nCT1pW7sAJKggb+DZyEisyl0jwK8fVttFraIdhqO2Ko";
            sout << "o2Mfc8+V5jVTIC/VPtuiK0Wjv0+Eov2U1UgvFM0jsGIPfVZV5UDra0rWjp/vnzo5C9MBy/MxfCyZ";
            sout << "5tC8AnEXpI6V5JSHSb8xRAHWZg4HUMmoZG9u4mX+fW9Lc1OET8xVMzfN68kndi25/bnfx6SpR6f1";
            sout << "30Nhy8rd93qy7DpYUHhLLyhBgujuJiT4scogc2iDNwP/Fsam9RR/EeVjKv8gwDHrDSbbUbRbzfUD";
            sout << "fM7oph68ce4/sQ0rCh1WbtPrSQWbNU225oOYZr1lgrfxlYnI6ROd1M8nWS623ofggs4Wfh5CMYXJ";
            sout << "OYe0XlSNliVnU1CX4MXuX8dIXaEmU2HUjfKWif6YAQY9eSLikPWVgEYvpthn6dur/TP1jC+W3a/Q";
            sout << "vigZwfynqIl/NC8Pe7tpm2qe/K6TcOgru5ojdaPeyLUsVx/wqvvVlT8ZOgVV2vKOXAUQ8bLXAH6N";
            sout << "kccB9Aw6NLtXi63VjdwUrgfBwT3orFckTuYA4u60vxs6e2ocbkJa8YsPyN71Q13O7t5q7aYP3O4H";
            sout << "JbDKncX6fZddc26LFxgRQem44nxcab0yoiP6H3cp8mHB86+vkBuSQHWGSWN5uNK03BP0rlUntMT8";
            sout << "zis2iDvOY2gDKYHPj57HszSQ8/SZx7MFKiFRTIKMK5P39lNz0ylKhskAfZ4KDzLp6xzwjwxDsy70";
            sout << "LQatqB5ojtaMcglatNYvdU48iHz7T/KPIU7fs/vySYdx4EHZ+Oei/dKFvpdK+y9lFyRJUoBXpb/L";
            sout << "LCs202cmGv01lonaN1Qe0QEL8OTPqvPwGb/rqsf6CNobSZd7mmiMwlQ4dARKyk3PWgYvalT7nm7w";
            sout << "aZvPlDUnws05FXtATiepdeNlffsCK+G/TEzH4vxOzbFsNwhwqL2tVf0t+1UJwD+NQaF4p0mSoA8B";
            sout << "TZCslymygnhgF9Tu0jHPigJ99Nj36RhdYkndHiry2OlwpTACuBv2CFv9F8SDhfwuRw4sr3xQpomb";
            sout << "vpTNyGrXXmSVrsTnGP506nCBg4IQOi5zXA6MgVNXiy62mh6u0lceyZXfbvCKkGJi6wUj05DdrCvY";
            sout << "3TrTH7DxcwTZ01O1e2A+6A4v7u5/GyN2vEQb/p+lSZ9XAWYLywIId2pFMJ6nDuD4HKuoVljrA4+A";
            sout << "wrwACO6QAn0H15I4nIBRzxgEhL+/U9Cmhmudtl5iKeZCifsTrIda80CJ9rTl/w6knmll4/9rLceK";
            sout << "QmMKBXjd4sdpcDh8YNAUa4lWNsgL7ElkbTzmZlPe9ScYHAfcJGEtwrOuYDp5LfJOh76nM/LbGAkF";
            sout << "qR596/Sqf23C+pfe4JH2lerREvbvkx+N9b6lRHEEU6tN5qx4AVPUlVDi2/1qo4wcIFJEKYS4bfKc";
            sout << "jZdriM9msBQPzO1GDoVTfUxcY7vVtkVSjVONgpjlM4Bsr8YdqtRH7xAEuMFG5y97X5PK6KkIAqAc";
            sout << "fXc8BSroGGBcXmIx5dAh/U6oQAu8gOe8mSvy84/dz2sFbioZSlVWR8asbFLDjrfHHpo4d4EX3t1D";
            sout << "XsGHOG0QPuj5/lH2QZerKkfyMZiD7rSJTnpf8oUyY+gw/e0aZMHk510G4ybRKgY+9tCGnu/A3cJc";
            sout << "5wi4HcD1PUKmZwKZI2PIfoIv/VUYQSZt7U26rf0wUZEMcuUldIxiU583HZ58fUGjmNhi5uWdDeIF";
            sout << "pgIvY8tvmwUNT3W7H3iw+idsfn1w1CjWmWiCRo+yjAB+TiOxHD4JCCNAcM5PCWDa/NXLBA2tO1h9";
            sout << "Tk8h96lBlAtu4IQVJaB6+o7iPk2GNq6FAiFQwYC2F/Oh5A3wKWrd2kqQ8JeZtNdk4L2+eHg3aDAU";
            sout << "4aXaYzfxDSOYHVHLgDj4VY6ulSZx9DBst5UPsHbaLaANf98uIZuji2hXOcg/FEnchb7A5bb+UIyV";
            sout << "jJH9d2D4R80e5GwPF/vVZK2GQr7h/pEgdGTh6hr8piMsR5fzJpEI9dS2jTJnet4FtnhDo90fImU7";
            sout << "O07J4AT+uO6k0B+lKuc5QlmaXg5Hm3hCqq5CggvnHNKWHV4f1ux0GNn19RiXwX18EdkxSEDIX1W9";
            sout << "CmdCn7gQtqznt3pNq3apLeITu3phYapWi3FlHhxoxOE8gKjF204FYOBeKztMqCqSlYa3ALQiac1W";
            sout << "bk5HyFaFvWEG5Rb2gs7bHSgHuDFlJwErbiDfHM8415fDkbWAOMJFTDh3YY/tuNz4vU4bNC7TcJHn";
            sout << "o5gak21470Iy5oZ9WHOgQm07tsFNxyJg/98rl1fQ0lcjNF+2nrhj8O3bXRpgt0eh9AQeu6QVTC9A";
            sout << "d7DzZeQ7N3HCWJl+VbwFCW8Jyo5umAkO5qFn+c15SO7QZ0KEjEORTiUnBDXzAtQjDHURr4yEjtiS";
            sout << "LkLTQsXwiwCmBOnoTdEPegCHIEeVSIye1t0x+dR2DeqK3WA/hB8mFawdIlhtmdEOZXov1mutXVDB";
            sout << "LBN80uBR/EypdQyctL5IR7IZIS8p1vG9kY2yceXlszfRZ/t1H3jx3sRSJEdVoMOP6Es4IpKj5uw4";
            sout << "agZ/POEGbn9qKdL1N2MM0R3y6D6YjIXhQF6QO3XsYlzU/OeOWxoemfC3/qn4yxwbaYyEWjICn/9d";
            sout << "e+z3q8F3Kh2JC5VsuIVaKZ7MqXxvtn9SBuUQ2vsx/VqH/LyRi9glQZom7H7XYefzQ6neF1DobC/p";
            sout << "HsFSf3Q8QV215kXq7lm0FUxQHcA2p4F037be3CN4V8aa4NqL/ChPqdR/EMtB4C72nOJClAGY0BsL";
            sout << "4ApuEZj/SfshLKcnMdTcJTsqBDrm2ykdu8JEg5PQT3vpG72eUbAm3Mbm3CYImISFQiZ07XYDnN6J";
            sout << "DUOjaiwfjJ4ydnkaAQbDnChIS6JXZV1LEMS19DCB0TLoRl6jnX3mNu4S+fugfFHqXYxgKIAen83P";
            sout << "YdaCotX2Q4qj7YjNhknIksW1OPA+8VxgBhrO/xBocrJY+uEi4+iVFJE/8Ge8sXmYr4AcY8q6/CuF";
            sout << "jYWYkmj8UZp69kE9QGJdiwmHBFOLIgAdnKMXuRS2NFZgFcoGMWMxeasH12WFGRmtuoS+eBi7RzUn";
            sout << "CC7RpHf++XNJClYpMSPnEYafiDQOcMmjHcaBYkEBB6kRV6dPtwrb0ZQPGVFYnhhcOtcM8tkLD0k6";
            sout << "Ek1+UYaG75go18OFlshMOzU3Rq7SuuNwtSNQ6drsWWcGYEOWtQG1b6DniQ4+e9hkP1sHI9GE3jMV";
            sout << "tbuSi8Q7pRY+vzyHyNIOH5FmnWCFPYLMkHt+6aenSR4bh9b8Bx7khop+XymNuNOZg/hu863yOx7P";
            sout << "D/OCXH8BflcACMxVMFKcPMxYZHrxC4cINAO2wY7ooge06LSlM/3gNNwfizSn5miDw+3d4UsnnD6G";
            sout << "+oMA9AJTuPcy0tMqBsGqmBZMAIO3yHWroF9G0dS2TVJqMn8Yt5shS7TKoVn2ognWInejfbOSP8nl";
            sout << "wliaBSd4XCMRoacNMgoPilfwQsqwntNn1jsNRVG3YwQwJnSWXo0YUvbCeijqzP0pX7IezmCY8qgI";
            sout << "VG7spKv3J900W4CCvbAlNsfHnNAnjU/5MWjs7I2j+WhuFv+b++3vDsedHI5nUWjTUYYy7N8O48LX";
            sout << "XH042vXLucXjT4inm+IWDpj+br+GmppCY/bZzqO/IwzMt8pWiReyU9NNtawN8ag6FYA9fxrqGwCf";
            sout << "j98DdzKzgBgo28R8Y2Al0oC52pFovY0Ym/ormPTSwtehSaq3lTFgbCKBhYR9QPfx3vMbYZupWWCD";
            sout << "FjnZYkKtJltasN+SKs7Wp/cCU+U+6zPAqHmv+ZSWiNhC4lPgYDGSdyhPhp/sHz9rW582bdo0iUT0";
            sout << "8QRPz1UGheorTDK0K+V0sR6ltaIS4/5P7/QzmVvGnnilWTFmqdBViBZRFv+3wp5xMqMjtZthbZtI";
            sout << "3YmEvxUwnPvJCu3+veeErSbvB9AKDj0PJGmqLx8Cw+SbQgN874XSSq+w4WVXCT+GOUrspzZR1LNi";
            sout << "59SSCEs6AKkKmXmjk7sdl6FocyINLIGBUSPc+nwCgESD4xSaR1cx2SjSf0CynPF/YprCxvqRvQEZ";
            sout << "dcKWvwgeU6oJP/0tz55yIZjiIi47ucY4ySXSel5iPoofoh4Gg21tguOQovIAbzqYrkN44bmBy+2S";
            sout << "Bpxa7JRdQr+QXwMMEbsnjrXbdHL7J5fZa/80FkwOHY8Fvz1wCRoZ5y0EyAZCq1s5H4Vd4yHMaZmW";
            sout << "ZNJzXvADrDSO7NxyTDGo4pVX4Dxh/v854SiHkO12eDvoizsraCz2ENtE6cbpHIdRq0Tt/K1HEH1E";
            sout << "V9l7eNcD6XJrvSe+6yBEJKD7srWbyC3MPSzfl7cFEZPbVzfkjSP8H7AnZl9mFnEv8AbuLvHJTVzq";
            sout << "LdwmDV35e4jFpDbf61+kzeO0F9ABrphKOY2YP01f0sKoIicMWOcXyBVIMCtB/9Bav/Pn6bH0pyn9";
            sout << "fyVa22J4yLzvKdYKluBaKUYVVUS5GnvTT1Qxdtlv0AWceJApOg4sHx9zZmoUMDlQQqH54S9sikKk";
            sout << "NnW4TD+CbdPYFH7kgXAkoFveAOIKcZZhTUD3FLbarLFqwcU37NSuqM841MlPW4Koa5Jrc5ySoL2U";
            sout << "FU/Xu07j/PvrpxdVHbok0aRtPgNm44+vPDAWmuG8zuxqluSW2xxzPi/YVo7nsWPvJoDKjZA1aRMr";
            sout << "LehSLFoQa66fCq39/XXbQp83pL9b3tFGRndFnP8FxvMcCaYkG/SDevUg+cG1ysNbBSBJ1plCJe4S";
            sout << "loGs/jQUQCNIX2/mR9L9PHM7FS+KfQn0tHgjipj5DRGiFInscy8cXn/44uYSHLvZUTvdcg3+I9hi";
            sout << "qdV0luQoUCZm9ooT/SBeEo4VzB6+L+fsuOlYCQloHNtmJ3KrHtRtbk3caJcrbTcoQETMbSGN5awK";
            sout << "XueIC5Ar7zL30I4/DTjrhho+/6Kf4nRhplqPpt+0lyLOSk9KjeNxWktMwnnplhggnxLqb9laXNAM";
            sout << "juhmMFkTHN21kH6F3TDmBE6rl8iMHHn1+kSKhhYjDeb+n35qu3kYYEg651ow0NVdq+aPC4YYCvt8";
            sout << "44xqYd6LxPeXF66jNdNl+oOBdds5MLOyW8uIFdQbmh514Q41qZ3TEMieqT08Rp7qW4/mqiPjRzp8";
            sout << "DZgOaUXJnQnUKkVbqqaiBv+ot+ArYe5JB+e7nvzgyf2zxCZLk9/Szpqn+JbGGEHFixV99XjJQRTU";
            sout << "q5DcOkzNjd2/Da4VEXCfjQPWgYlYu1u4hSgasO4GVpGFnyuj76gnzPGbkSVq5GMJ4KzYtl7/sta6";
            sout << "sijpcScgVlvvL6ff7fhEVnwQfa4mCZn9/umsDB5ZbG+VJufprMVSb5CMYcMyOU2KYDyvWHE6jSP3";
            sout << "Za1W9Fjo1Fkzx99+pV5Hex/GeiEOxyYCPjNEvCNpn9xJWjU8fx6/6BbN2mGET9hO18lol0OFeBio";
            sout << "88h5F6msca4plFbxu7eLv1Mx4kyQNmvupkPaufis95NgiPhNwOUSPsGffK8WGIaJJx7+g/SkgN1h";
            sout << "RqaZGgxZdrnY9nOOB4TsqNd1dG4p8ybarjysGHg0JQEFRcxj22DlC2PoAMEEILAjYb3cIX0xxy18";
            sout << "D6TPca+XaHi6LoYixk1zz+S8FdUGweRHS43IaWzBa9KDJ5vEIxAAXpyJ5Uv5PRcBhAfdjqjeT8DN";
            sout << "bVN0R0D5QQA=";

            // Put the data into the istream sin
            sin.str(sout.str());
            sout.str("");

            // Decode the base64 text into its compressed binary form
            base64_coder.decode(sin, sout);
            sin.clear();
            sin.str(sout.str());
            sout.str("");

            // Decompress the data into its original form
            compressor.decompress(sin, sout);

            // Return the decoded and decompressed data
            return sout.str();
        }

        // ------------------------------------------------------------------------------------

        // This function returns the contents of the file 'frame_000102.bmp'
        static const std::string get_decoded_string_frame_000102()
        {
            dlib::base64 base64_coder;
            dlib::compress_stream::kernel_1ea compressor;
            std::ostringstream sout;
            std::istringstream sin;

            // The base64 encoded data from the file '..\..\examples\video_frames\frame_000102.bmp' we want to decode and return.
            sout << "Qld+lmlZhXVX5NDRWIFG1T4+OGdJbkmxAXXdHZEtpDf6knVTlRWyAhgv85Tf11KJZbhKv1dcsKtQ";
            sout << "fX3y/9RxoGwtXxwFXCxbBrLRaklamboDoK5N06vzzd+CS5m2gEc6/c0nITY0hfeRURvJI6cvb8OM";
            sout << "W3UbsFYtU1ntq1umGkQDDu2ukl7sV9LfHj5eh+n3zjy7VIX/RAghwQA44rcpnh2d9icwDh3G2dCo";
            sout << "KxHo3W4qqxVLis7IIGg0dzwguoB8ZGmlvOg7hdMz20beAel/LlU8TMghQk2Aj+4NTKQRVwcfXfdd";
            sout << "AyOhcT9wYaLabnDZgL+7yq8Ji3VHvlGqgVD14WJj87s83dgMGikK03QgTzhyY4dwADT1aQ+xqeec";
            sout << "K5d37u4xE+btNpcqFhHx1YCp1r6+O9kjWRE08rfb1qWZB4qyCwTkxc1C5buXaeNRrw6pHHqSLWd0";
            sout << "9IFAbB0ukrEWmdHr8lVKAfBzL+fTFrAO8gjgT/vf15asPfqSIl+zJyHK+YJ1p0F/UwIzPkKl5nx2";
            sout << "O3u06yBtIHKdflKGjRSXomDYGjrjeoLNjqF//7zBBXyNXXv9TsWwekzsX7A+qRdm/j/C0lCHU+sL";
            sout << "g6wWh7L0kL4ARqClFQfanO+87/EPhnu9LuovQzTY8PhLB7a0R2fhAncpTEPCHBm/0bTEL5F7V+W3";
            sout << "8yzUfcgi1NOCAUBN1tVwWBDsVv/NUIl1x+FTKd1BFAsN7a0LLZO3IGnCikPyU/Tn+TCslyvvnU7C";
            sout << "SxgTwVZIeDC4sq0l7kk3Jc3UWrDI77EQHjxNFKG+lnJxMLXrNMy/lJ621WGz8uVPvpkcTajH/YdN";
            sout << "eaRRgbdqnkqmoSyX7cJeaZG4NE0n+isv3j7WU3cM6N8WKhYCH0ReG7XjbdBgrPHaUdFD1dV20eC1";
            sout << "2PSv5D2jEJkDUVNdMZSmwOtF/UDg4WRMY80YLGSKhxqLqoszhKWXltFdm1M2+WPtSFOdr4i3wZRs";
            sout << "W1At1+bmMTFCM18KnBg0mQv6zyjKRjzBPHLiQZQVBmv3al6bQnyLsbpNDM79oVd4cdEv+l6Cze9Q";
            sout << "gPwotlVgN8hi9gcF3lH+KTQ3Nmd7WK3+125inZjFasNLUVJZNLshz0DVnKWeaO4RGLx/Nb2ceMwG";
            sout << "YknvrhsARhzLZe8ZFcdvQs8JnlXwuNpN9OKtBNQMpQHf6ytCWCV/ZCsrFCT+GnKMQxCvR4/LNIFU";
            sout << "vWKvSIe1iIDoOhUCgJknwUH1M1Ug/VLDpOxT4ylapmV+YoKSYnJANOQ3+3FhqHRc2KG8vGMTNzab";
            sout << "3rs5TEQVOdP+fEtU8oynfcb/0WK1BAKgmDRhKcz8XG28EX15aWyr6uJkAxqYE8UXHOr+kn27TuWA";
            sout << "49KqCFNsNcjhvnjWQEBvwTwGRKofE4ASdnQi/fF6Ozk321H0GJYpXETW69XJiNzT0f6lDUYWR+Bd";
            sout << "HC1sW256R9dn/MBAcSH2FRUf0Mv7z0yhkb4aUwHNvGhHPX5g0Bhy6x1TCOqHtjubkHkxcq6YpwW0";
            sout << "f+9DXG2M/sK+IPqqgy4tY/fcJeYxq3Vd4+voDPYfVxcWpmy/4X5kzf+fFwtXEu2MBdYKQEKviV0Y";
            sout << "WixgGguhWig6d2vAbUKgAlJd9yixtBum62nEK1vMZmwzfokP80fimNdw7Wony6EA+v1oB+zHQojG";
            sout << "1fNaBTacRa+NsW58U848DFrsJ94S/9ss/RUx8GuaWwKP6WImvkhyHP+YdbtkykVPVzm3s798VWAX";
            sout << "nvjz9R2NlupUdBVgJg03BAH4DoJdYWOxnLtUChjW++StA7OGgWoay7DDs/6AJYycFh+TFW8D/HhA";
            sout << "sEQCUAYDnpze0aSX6CBZ7vHMr4z9Gr6V1zE4dbJ27Yv7ZFBetDLTN3MN1vk8SYDFeFHtzyMBfqor";
            sout << "hHXu4rKtGW2j084a3BQvlDVZ5bQX8mh6QBENJ+C3kBUhE4QthRIUwGsuarKCnOUAAL2Zt64CgGKi";
            sout << "gRF4wZ3V3Tt/FZ725pUJe2Td65CddbBd2mRqEzdN+qqn24au9ucznuEr7E2j0T3ey1RWEcU7Xsx9";
            sout << "dEX6LoyFqowIEvyGOwdTlycOq4v53Z2cJTaitjGSvtaL1h1ASDW11jvsX0OHueMJJyLFwKOjj5JL";
            sout << "JrtfkGvWRrycw3UdT6g2n8IPKk8j96hds+UlDvgwIzmlromWvPfwixA9mt8MRqJlBIo94lKjOvuw";
            sout << "sS5FUcr31Dqmaz7uQcrwmxh35jAvW3EbMObKNdbaojaZl3u2Szt4LA0yFs7UJpkkJgEO9E7lGte3";
            sout << "WAUC09lpDNq1CLvEfOEsWQqzXayT9AU86CCnFgi6iErKoA3uNToSaqQVOxOi8RWzQWuNfmkluBMF";
            sout << "6NmgipWLj1nlLLBUOpQD6nVlwYqZEeiaQOhHRYUiSUFbJ1EVdBPoRi9CPOquM1WX5H9qbUOBchB1";
            sout << "/oI2wsh86ZqOld75VfD5QsHXmrFsQ74mO4MyvAojodFWADFmRloe6UoEBFYhjXSlj2sUSTyAMfPh";
            sout << "jgRrJwdMb0/53hm/Emqz9AhG9FpTI5P6doDAm19dGp09NvQyYoZnCCWHC6fPpl+BS9WANNfsSZnl";
            sout << "dtI/JmBbtlx51QDTA+Dc/RqGZZFzIIDnBoIUnyFY6FBb/S03rQadSUdeuBInghBu3bw1UH0AImFZ";
            sout << "uy0gKklA51AkE3xVxtyGaCniGugrWwcGYl+FeiisBNKtZtbo3spTO/Rr2Fjg/mbJ1I0dCDf2gJt5";
            sout << "qkjF7zRl35+gl58FjkBxMdzM64289w8wqjKO6P1JBnJV2gsbU6QYGN/w3Xy9BC6Yq7WaHU2eUves";
            sout << "tyuWAxa2+pl8SzrrvubhGUV0Rx4wVcRCLilUKHQ9VoCEJeou1y15h/hl8BkmuU65Do4a5txsH+Xt";
            sout << "5zVkkmGnvrvq8xMd15UIz867U7lt+BwgV8UoDz5ZQnQXvwt4+NLK8dLaRfr3Tb7KpLPOdymoOW1H";
            sout << "7ZnfmR75beprkaMn29qygGhlTmVTxNC7CppNNfS97Yc3ki+g14rkFZMmIpETfUnM1eqp9O1+y5Lt";
            sout << "1p1DrkrZduSwZtg4DmJt/2g0eV5xO7LPishn8ezxLTcMLCQYZDM4gjhiWt26U0XEQClxxouxQKiV";
            sout << "jpcwn99nBEj5r7tkQLfJ4clQW8gOuBidgJ3gWiAlR/Pwkys4L7cdQybo8WegBvXRJyNi0mDGsM4k";
            sout << "cD8A38+zxN/mI7Z1Gv7y5zWUMrILQJvPVXCCWXo6gikdJpkLrprcp0RUXo0b1iUgGgOXVvY6PGA9";
            sout << "HrM5/u0t40+Yh7+3XqzLWWU9kuOgDLvjwYeuRu7oniMmTWOuRtkWyqKmTXlrwjBz+yqIQJyiAa0h";
            sout << "NF+n6CNX4Bk1X2h7DIbZf/hCbYAsEQ/zHAudSiEfzew7T8r/h4YG2huCTjV6NrBNOzxk2Bu2A2kd";
            sout << "PbIOFbLWphpDX3pQtfSvx46IL0uX4O03VgTiz2NCvNKaH6oZZ8mTxkV7QmVnRdHG2my+oFsqtkv0";
            sout << "uRF3h9xXKDij2r3lfNRhHrM3OAsySepTuXa0hN2bU0DIiX+BlxVjBKicOLIQUI2k9ws3akGMpicU";
            sout << "vO/fKIwOs7D5L2oUOqDcycceYM3DCTjyjdeP9hq9+w3nxnuvtoNuFnE67nR0FA/XSYLjZOxq5EYD";
            sout << "N4LdNhDWzYFjEHXgmyvc6IrCcltsisuRN1GNZmXDTd++/8CdA9ThbqH/AZvMeUt3zZeKMmb4Ad3o";
            sout << "T9in73sligZv9ed32/6rc3w96aoiGtHgv5szIJ+h49hCNfuVoSR6GB89zKr3cHcsQ1PyI9+S7ON/";
            sout << "/lTAQtwIUSGSQxM7xmawtKVSit1zv/gw307SfoWAZMmz3TzrMzKbe3mmNNwW91a6GLqe1iyaQ9Lf";
            sout << "KdIVsCHzqq3DKgex8X0XPnP699SlqSR9gjHWZvuVOzZkA5WE8oatWQNS/W/bhBaaAiT5/781O1s2";
            sout << "9y+RfF/lQh6tQbhPthbY/ALQDR07LM43qqvdaiinxT1gYh3pJJHs6oEQEAZwNvbjUZURmwbJ+hE7";
            sout << "fJnWbYtftZn7pxlJXzXueWe7IPmYuzrn6jJEh4xbnANe2gpzkGPi2mHIlJ8ij9vp5KwkaZ+SkmEB";
            sout << "El1hUdhsJoXXErzKm76RD0oLriVUGd2yvI/pM5Mm/3/pykt8cARjatq2CVc2YE8P2M0cC3ItKD0Z";
            sout << "YqnMdxmJpxys5JoN7q9yemZFpiV2OMJ24kPc8BGJtgbVQBWpMUnNEOZOI0P17D3QRehHWf543/X5";
            sout << "PqzfHj9J1tUcJMNCpp3GFxejtpOvKD8J7JB7+0QzqCCOE8SQIC3Bs6IB3IeZiYj70f3XU1UPF028";
            sout << "Kbkis0YuvkL79pWPoqNAoA7YVVGhnb7ed4p2/lqFw1NHG0+H4YQO1gz8U6MeiUrjcxA/IkwNFAZX";
            sout << "5WsC2QSadmv7OtUT6L9ifTCs0sogqcC5UETke+cLIfprQ/c4qIMP6JWMqRasR4+0qjxZbSxovEme";
            sout << "0oIBoRGE2za5L0/Ulwcx4URU5/iUu9u7IPEcUVQzABaMIN4wMA4Lbd2evs1Xjo6xcFLK/DBtOUdM";
            sout << "JTmiOUwloSGDd8MSx25d3BM1HdXfUc5uYihDFkk/0AUSLa3x0GbKimowJYGAkr6Px/5MsCnFW+Ix";
            sout << "anv9LFErILEMTwCvETg2LuRDSC/2uyGNnndivYpOu8QGplJbW9n4ALWpvS/dc6PJKjZo6GhqqK/X";
            sout << "U8Q39jn5hSYIGb/LXS0mMJZ7CPTO1AcNCAZb2r7Fc3orKeMoBDA+9R2x+ZQnlGWV8PdAzfpnfsIa";
            sout << "O+JL5Nh2NimtPQsqEVykbF1RF3caiYPCcEj+SAUFSDZ2pe+YfXPOpe2EOZ/BnGTQWdn8u87mw68G";
            sout << "Y2zcXQAjUHXuu8iF3IXR0bbRR4bDKp1MJ3YW3nlk2E37v0SEkyX7EpXg8hvNB0uReBLOk8pxw4Ha";
            sout << "8R6ekjMAfMnOQynT0gi3tOoy6/CNXBQFAp7/u+Owz+DO+pRX9StkBjFUWe6+8ZeG9YJLnNJZUGkY";
            sout << "Ki0EEoWpBbW4zvGa0pOPJ/OceaYIzhL2/1DohO4dO9jJkOU6UxpYaHMOqxr56K7MCpJF6k3ii/q5";
            sout << "yLDM1Ee3peTb6MfWrrkeYc9PhEtD6TdlJwQpHkivrFvBvp5tz4Tg2zoW4O27SZZn9Hr0MPAKCT9E";
            sout << "Vouc/LBstfyV8cwXm+nqNGg+f9ZLgns7No8isMXQTPso65SeRw1K48aIwSqeZ/LZ3fQZ+n8l0M0B";
            sout << "BSRAF11nGpSNKY/G+7vKuJ+FVV1z1MXmc9wp4tjnYRMEBsmhTnL9fFpoHvDlOAgI7ZN+YTKFOP8R";
            sout << "RzeXkz6ne3/ztgWIS6QZ1xFzl3rEPwvcADeMAT5C/j9COs8YXQUafQ4o2I42PM8PloMpz73nTNqR";
            sout << "yot/tF65hOa7X6sYFISZxTXvAq1qaV4j/bOyv7Rr4IMQkvPhZRnKuOKd1lEtEleREDKdNA+3Un3u";
            sout << "3czlYhhMp358CRLKGyMNsLc0oVmrc3Zf7AljUJg3j/WmyZdSWCfCFAUzh4y+OF0r0LSxy566EC0f";
            sout << "EvoeoqDl6Dn20my8VVm7Ek7kqtjGX8OPXh+OYw7q7GDHCLn+gSQKRPZEMAVwFPfY5J0qUdn2EhBi";
            sout << "bIdpDLJlX5A9Tuvsfk6/TgWJWrd1ljnk5LRU1JdLpy9Gr3i7RuUzHVmxutyDezdNhJQ2a/dGJoyL";
            sout << "wqtRLh+sAldKv7SyftQMUr3mVgcuViK24HOW55optbEe8LiDnZVyjx0fWPnvUF3jrOu9iKpCzEQb";
            sout << "UcV/ItitPx76PWI/HUsXW9nPkyHBkVhy5n4A5OFbK4O7r1eiQBRttalZs9UPIX/zGjhL+mKsmDPZ";
            sout << "5W1ejsVxJT54FOwFmCD6yxgeWyGirMHJ8iIwL12Vf2WG3NvnLTIMokCOIrJIGl2nsgpYWFh66aLF";
            sout << "BY52VplzlZ7uvieN8T1gFkoD1GKDTPvxJdbUIyCQG09lZhFsqWDT1h+sZMk2cE4UGFa9SUZY8cC3";
            sout << "n8nDCxxmmQybJp70Ig8cQPy9US51O+PL8ZjyRJirlSwqhSXndjyvgm4xk9uOSk7AooavjRJYlO7r";
            sout << "Mm+U11XEN5mZJL10W3nMa3Ls99DQIrifOcWVAaY0pEjmeBWHizxOwwqt3X9CY+Sa/X35awGgaQCe";
            sout << "CaaIXBkxwkRLD8/Of5fYQ8Dwg+LWZnkMjs1IBdauSjdWvT7B9CXK7O+YJC5kjduIGOAu8qfJqG6o";
            sout << "HP/uOEvCq4vT6Fg/gLbOjNenVhZSq6bAVSgG3C8/WonUeNQiWnEI0xLm+WOU1L6bcqBNHNBOOiOB";
            sout << "yn1YojmgQszuBRj2yC4rCAkddymD/yzJ99g8QmhpzlakTRtpjf9LTErhIBDIQSd19U8mNdu1u+cT";
            sout << "PeIl7vlVmhhnojOHfLGRvD5Zy5dC9XZiXRRcNWHj9vOTdrmtwy0mPIZZMPSBLJnur3djgCLfMniS";
            sout << "Q8iBSo8Uzf00zSZBUBUmTekAAHFvf+we0/ERcXVkJ2ikfZ0v3sxGf4zIVGAiI0Nx6PZTWgW1xH9a";
            sout << "3OYLo3mzzz7TVy/Yueh2awJyZ6CWq/ysu9x3tSJG8DbBBc+4segNq+IKeu52mYTcLmASIR9PT12z";
            sout << "ifAJWWtKNPR4T/FK1pDTkaVkULRPzqHP8jxerpBRMnt6FStshZiZOs+DZVrXJxSP64ZxFthgqMd2";
            sout << "UvzcqeI8yD8ndIyd69Os5t9OoeWseLPWq95+5OUX5v13wiS+EHbJBNnZLLt4K8udeohSNUXEBDND";
            sout << "tj7TDkzLyXYpcm3KYkJgtRY88kwidchhjvMQRDNw15ozLnR01t7UsNLVLuBij9sUBDA81i4YXyrr";
            sout << "5se81KvEZG5kLLkFCUnPKRU/3aRb40F3zvY9pcBU6KeJsjBQzKRX/EqFise10qyClgAyvADqyW8P";
            sout << "jMDslvAq1mY3kJJU74AQNRuFP0iBJWTX5B3BinqVoE+VJzZPbziFVs0riZccQIoff2DiHS49f06c";
            sout << "NvWgHb04xNXz1skoINdhFVRSsjik+qmxbFE90F+h9eshTMxwQVobiRLdLv1pGkVOpsRxl0eDsPMI";
            sout << "NECgDvYSyRtSxXz+SUGk4dokSgkrTPa6NDx82FYiyITDu7wcgtQBOTj+SXRWrrG+KB1MR5cRHWP6";
            sout << "nmELArZ2JquWF90mbylFHbzKZNB27/TLN3X4/0cVqLNxyzow44+Z8f26lOFFnh9qfzqS874839gB";
            sout << "IRvAiKnuSo7KCh9BbsvAHk8a23Ei/KNxJFH745cvLab2oVcPuFdwsDvuYgPNT22tuLb+/QN7djB+";
            sout << "6a84/73zWFCnfeMlPgtGbxUE9yns4nYTNx+jLDUpKAmYBWeiPEiGhsvJYwy/dKvXF+CVdUHFbzJO";
            sout << "r6rnCYGBY4Urxy+S6KPldLloAnA2SgsqfiU3B59g18OuY2NyPQ2fj5/Ytpum62hQo3SlZTVGXFq7";
            sout << "pInYYEueHtHIyJA7xpqX7TG8HrdRJRbz0Obs5X0P1vbpELhaaI0vppEIcfzZrmrqQfOT9fUILia4";
            sout << "DOCR6e6TWVirmzFa8hilcMLjG89m/+h34mO+pwqNsURvWoJ7oixSe9arMPUl2pKyA/wICP8AjQak";
            sout << "XXaacTp0DMIbcOsf1NY14cuKUgl7tcst714GMXCHfPCnYMUzoawg4o6LVOzHwHRgys0aWz4aNmQq";
            sout << "5UJDqc1UWVweo8LD1++5BCSMO2pNJaKdbHqVhT9yJP5NxIkV1UxvH1TowNN4SI7q7BMH2fxRBB5e";
            sout << "bu2XU2s/5Vl7snSkKk0PxmbiiwWGiN1Dx9YIUCEZGJcomUh7phIlPaoT44MEDMT96z7neW110We6";
            sout << "/FUCAtq2aeF887UH/ARplwM/elwZyI2BIDGAWF5udKyRtCrVznJdXgtlNND4a8MM2OQicC3D85Z4";
            sout << "suz+m7cwYIW1o0xBcuM2ASLhhAdATCHoyD7A8fIm3dD0vMKM7vA8QQi8ETjiN7GzRqoyHOLt1+4U";
            sout << "wM/BQdskv+ufGD3KADK5CrKPyKzaosnPyGYyzclqSBbt//QUdYBQO4AP2A3hqh/QySkGID1r4fUC";
            sout << "cM+fXd4DhwvNjAZHRPfGPD2xPguPOmmuCxkHU1CpcKdBSbxR2q4FTdaQ+1HBbU6KZ6358kqT+5by";
            sout << "3LP0eS7VemMfP0d9rshtgbK/+nG30EGTtSIJe74inTJ63k3mkHb7jRu0rn+vmO2aO60HQM4SUiK5";
            sout << "eKZUz9LGB6lD49IqbEwj55SpSUmLsITyO9HdZ2hhAAxRTQeMz2xatUny/fFOGSUiBSeYvyrmnWjs";
            sout << "R93uXWixgiS7DC15ARLHAkvkhT76lGJkxfdyhdkneQlImBYUecqxHmGRAEOQEUFB7o7ARObjdaFr";
            sout << "Cii//3Mj+6/YSOAaYlS1YInt2Wo2+gin8+eeqpBp88j4NIH7qyKiGw/fco3w0Yi/x/2tbc+OR79D";
            sout << "HylA5tAU38lKI9ZuW3/8nYGw1mXE133YDK8ZVzGsU5JnTnCuIW1+aVLf5pzzldfg3z49X2S11QnW";
            sout << "SNtd3gZ6cY2sWHnQ0gm/EIQCF8696kwZlp/J7fZonWd5Fr3P57QepBjktJs2ReoUXGQGAYABNJAK";
            sout << "VdbPpCSu/kWlizZFudp+2MxUJnGMg72fGFn2AYhlW47Luuu3/DNQiHhmrxaqXbnpqk+jn52j2rQT";
            sout << "BRiIujM4aCQk8bXqFCvgD2pcFpKMKtwgkKeLlLMhqX4H4kKZ2h2KpJPSHsTg4EW6p0p1boVMoD8H";
            sout << "ZFZWTuDKtJOorF/hCTAC7paeeY6EqeEs1S+5ujoI2XjTNcai4TMAOe/pAdgpVIRIybPZP8kWAc44";
            sout << "P56yfuv2RYxcqAvmpEjvPqhaP0lG+bOUzxAelJy0hu0xe14CdvwWZseSDq+mDaFrrM2fEqGp0eBw";
            sout << "ysDRYDp0HJaY+QM7mS1chGIw3iTOg5IfI+wgz4BqRz6ILSX1Ci93QzzMsr+t8AqDssxzDX9kMNsW";
            sout << "oSoRYt29qfxPr7wjBdWURj2KxpW5yCygTMfAh53XuqjyEmuiSu30o3fCOzdQFOppq/LUul6A/Xbj";
            sout << "anZBHBWlmaIpPtvtdvjcWrflDrwdsJVZENu3qKyEmY+wpE2MMqXUtctJRBr1eauRU/Kcn+3JfSji";
            sout << "PtgwJ2W6ztTzjqr+6+TRCqJ7xnzIZ2Cj9jxjcD5VCEcqAVg1/rlP50cHMlBrW+UIySQQRG1FtX7k";
            sout << "8zqh30BJhM6Qff6NrvLkpzAl6qKkgcSe3tsbmovCf5EmB0irBvKg15LiCDxI0HyYT+KfKv5NBgjB";
            sout << "NQDDNZzqulGPMIqzBGaZc29SfIHaMxYihQS5mONL345HTVNQg7MkcHIYalGwbpIbq3GKv5MumOVj";
            sout << "SP1kZ2BGFt2U1A0ytDfQLhHan2a7v4+7T5JgUkRrvP8hKHRm0JTMFqFxZpzGOEQ9dwl6atFoNdQw";
            sout << "f+5ZTWFfzYzWxfvZT0j18yQ4fo8VQzCqlOc5kLYatapswTogEQ1PGd3UH9gTs4SfMjmvrfre/kBL";
            sout << "nTISbSjaS3Z7N81jEQkExFHqRfxiMIK+h5VRZ/FwviueAATnsIY6vChZYueN9Y8m92Ou81EHj6Je";
            sout << "o4mKrrtkUETSWP7IkTd4wD/EC7hMWvA7c9cWpY7Q0nAcruSrfcUgks3mW//cNgLmMo0UE4+0W8LM";
            sout << "H5Ib3drj/Ha1OAZK3NRhtUN2TRgJEaLWrvJ3fo/xDyIdmm/Ap0jf6jXGlXtdWl7KaMElGsmdnFyb";
            sout << "xgIsGgh9k6hyZ2uTFcTzeRFcsAVOBkQWOBhyqtqh2asmPFjOro/sNvstEVD4+SrHSPG8rlqK3nr6";
            sout << "SzNq0Is07At+PS14wWqp9ZseQSLr/pWwK7a/CyyAwxnbdJqhhsnxmODTz7nIWFgQittN2Z5l0RhP";
            sout << "C06ZoojtRQVQPyjCr8C0BnFGyKapnGPGWVbuzlu6xMhqC9C4T+4zjsl/hVT5BaBKvGgjvxUJ9pWi";
            sout << "fs120RVW4ra/T6X1oAXvYrjp+2i8QVcOPAC/jfE3zT1UpsTrt4IKYR1ILwbWsOi9ORJUqpQgQKpb";
            sout << "ZrMseH1LAmp1RzFN7M3Xo+wWx1fu9S0W3ZYizaVrkbWrGJyv99zuiCeIY8Ns7oqUfQAfBbJM2TPb";
            sout << "6jmkCyD1U5a1tir6vGc3YDkBryvw1oXRdeQEuKPSF3YVsp7bHf03ALIn5glbDJwh+8FvJVX0rkPa";
            sout << "yK4n0dU6mBDfyCt8yZ7/uFqPHVr9Y3d7Ec6RJPEouC3MOlKhYzow/lkVnCVteEaGqhGh42TsKnHt";
            sout << "I4MUSgA6n2QVTcFfD/hsQdtjzWoFupA0v3PsXobN27vz/aEGrtHjaPXKNDLfbRyW0OQN1Rfj/nRG";
            sout << "+TbEr9Y17f87ar0qnd0aLd/hgEJ65lW41HWWj3bjRP/RL/X5HpKVzj5zOPBfnOH04vCT2XtjAK2o";
            sout << "J9gFPdwz+6hWrKXGuQlrIJS+7RpVwZG4woY4cgBv95rFjfWwLrLuuX7PYeCJNLFZKpoAz3WpzRYQ";
            sout << "PepcF0/AzD6U/dLKAaI3pr/g3kItwzrhGIOz8ZLN4IrVYrSQ+kw+R9NuRWq6Wxg84hoRBWN3hfxt";
            sout << "A/58J+s/+DnmBilHfrYMhysYMVe0TaN0fM/Am5VIQW+lQJOVY8nVFqxxxt3BTNDAQtVl1BwaRQw1";
            sout << "7PVWmpFsBKX/cIkJRfuUo47InS4SvzV6JEgNSJb4Jp7BHtNHQpwKDkCjfwcajIrde2nIEXCZ7CPX";
            sout << "6TmNL3i/ys3IU0zW7v3uDZja60mepUlqvb3mJsiPNu0OkX26rIO+K/E9roqWnwp4HdzTCgv4Lmuz";
            sout << "5jaVBx3ZVxdjB0Eb3Kkt96pZflyLdI6WSR5RVCfpl8ov/QOKAjZoihm3/YZJJvytsUsj+Wi9CS3G";
            sout << "If3aps72fkcCfzlzB/2H7HS2Xmfor9P+hBdfEjBiz0oMlkXo0+JuamAk7ueQ5RO7YpnI61Q190x4";
            sout << "vgeGRm588X3qNb5IXuhumYUZJ1ATmHc9mMkNSNWEGdFy13DJmhHl+fOrADwMAy2+J8g0L2yHloaO";
            sout << "rSycaN7DgGe0YJD8IUbgS8QMOz8Kq1/Dy2/Uinix2smPobySnP+uQjRJ6ilbHnZCAa5t9KB1fNRs";
            sout << "HExRV0YMN0tZ/njB/hUsI+N/sg3lvPtSOOpXb8wAQQIcj5v7rCaCBbiGdfjWcgUbHfLDgFLTxq2q";
            sout << "LND3J3HmONGoE7kwPwNRBMD06KUY9PYwSqjQ7spkdBOYtuBVqCbnnaT6hnh/DsSTqG6DkymArX6G";
            sout << "Y4SesPr5KhQtUvkrHHQAAmQDWALzA7w0fKLi2unHSsRmteT1ctL4lGwVdEb/YFmNzxiD7BM3Zqvq";
            sout << "JtEDUScoB94K8R2BBOy/CQqEcln+DCoukLW8BdLAJ6b3mDaw8tLIevYutQwrQln0pc6HwXKmVTeb";
            sout << "AKKBMmZaAAClk5amuEkyI5YPen9gSZDX45yRohZcef7NSVOmI1ma3mc0rFYJYkJkC7T9bcpcHqAb";
            sout << "WONeENewbaLz1onBbrVYBdqPeBWjrGt4kOi+YSXtjm6WdZRtLf0cy2uRq1gvzg5oNIDGer12uAAA";
            sout << "AIqr189l1IrG/3MtrYrKK/Km9wuukr7dJe5Qf/96FXuft+BUsbt8WYQKOxeMZJgWfoZVUzKwkzEN";
            sout << "/+uWsF3fYne2EclBulQem6566qaX95eQMQv+mzrBVL7+PG6lweDQ6r6ZETzX5ogxLqyeqLSue8qJ";
            sout << "xmVPMI9fr4rDFcRrt5XgwH2j/QB8ymhsIsqPVnZKbq9+jrPUoRJDs1X6VCycaMS9bG36PFJwxW9y";
            sout << "Be+LAmVi2N4wlBCmNwXclqK1URJRvocKWAKrt0Kn4dILU1Y7hXNPjcUx/0oG5Ffoh0f12glRsyva";
            sout << "FJjtV7ePY1hClgBK9v7owd1YZUz9e10XcWOzS6L8HOpX9pocEiBdfO/oTDYBPd6LNk/E5V032Usw";
            sout << "XdqReOtJL/zOBPW1vJ5xc1UMRyGxDeqUQFxCKLkFAQj46cG89dM3bhj+Dh4U5J5MPhQltLLhIVbZ";
            sout << "qHvKkLr8cCYqSS5h/SOF9HWJKjNfXOgpRbR2l3/m8t71h6sSi5XR5o0NCi5V3/VCgAs/CfXCYH/O";
            sout << "WEDkfZKw1IgO15FUghj5y1vzejXv1logti/MHvh3WqMlQ5i6B6SgaGmIbOdqNHdYUbQnNg41m+8u";
            sout << "k5VPCiPWtGkbzzuzaYSx/Na14VdvAbZXDztFPbN3l1fMrAolDKlexOXohzVZhYOsqyu0xdGka67Q";
            sout << "i/Bz6DbYM/U4rF1lWoX1Xh9eslMlBlJjaAPdicvaJF7YArjzLKOHB9p+EZ5tfrotIKGfdGIbt/Yt";
            sout << "MSsyTx9PBINYD1u/JeI4SqXTwvWcfF8+ZmnspgnEabm8NKufPXhaJODzrxL7filfb0/JbL6qmf/b";
            sout << "5ksG2XT2sPxBM3TjQE3DAtnS/Y3psQUGsXUuY8oGNF/PgRWd8e+Ek9IoAnMLuvW7S4D4MP/wmWTX";
            sout << "kZzlpZBnBBvlh7H5yDHGYCwGUofjCiOk+4hU1LUNU4LGgZYUsfjNW+zbj81/JSoK8AMlmB/ERH+S";
            sout << "yuBnOIDJNY7BG1n0q/YoDIU/YuM09fyzW4tAHNqfbxaYif2QldKZWbxUEiiOWaItpW4QTsQe0MaC";
            sout << "h6ppkigA6/7DUgUy3PKon+JB/jdrUdW/2Q1qgLqV22zBLHJoq76n4QYZFmpLN7FkLFtabiXZigdm";
            sout << "7y4GEh33CQIlETdes59sn/0qIUj36ICMh1T/ujQOe8hlG1EIruBdZk9CtprIsVYdpuiftqJHRLmz";
            sout << "271Lti8HdM/KSVqrupdYb1GO00XOGVU4/kujCa/nzhCukpCpESV65sKSuIaW+VQlvYFATzX74WWL";
            sout << "bKfIvaa0TLR8pMz30JVDT7f+I568gRLEMsBdtMnLyJ8bgkPGdngpnz4G9FBOqkDeZdb/Ji0aKN8v";
            sout << "gwRZxrwUrjBE4EjfcRQy886hvReIP6GvKax4bwVcGfmYGdNuliSsCsPC6pLXSH3fnmYBHD8DbGct";
            sout << "Q7dlu1Z/aFTO/uY3yup3clLe89WTrPTK/N1lsUDULj/gPNwbWY5l3ay7YRoTFt49EjYPmx6wz3kc";
            sout << "YXYzOroiVWGspN9WfmHL4XN6pXWhiCOrMRG2Z43vbdUnqNk1992xy1q/ZHTw2ikxIwKy77wygWfx";
            sout << "28l1Iqj1k8Km2Ci8sBy/QCLnql6olazxv2fxn6vj+4QNrOcJb3sDuLbEm6BkUHIoiFzZEpc5nIpw";
            sout << "5YBSARDuvqMt+01+ed7yBUTms08EFh6EwPIsJ5R/6YE/Qt1aP/hfB7nGWlpGb+yqycdZbUw5XyZm";
            sout << "pOEHQHvW1j30bk27jCRHzktRYXRXQ3BpP+gVcAlXgxeoARjFfC3hYPFtKSQsmKpGKOFFMdvdO+qI";
            sout << "lwrru0fgpd0ctBiPsWR4loLRLpt/GHedyVRF29GtPPfP+JekSMDzQj4/73RZfIdHlc2b7zX+JdLz";
            sout << "Jk/4yNb2NmUnum55ezFqq7sv4HPFvMBuOth14QPJsrT1f0NyUBP3qYocsTAY7yo/fg2gYqLFMBHK";
            sout << "+N9IvtHqxyn4actl6t8DIHeBTsM48AJVza+LkOrmuK6k1x61G5r/qyEVTbWVMqKagbySMYtIyJxd";
            sout << "G87XVnI5bbsfSdTdTSQ2EOry9U2H3HIrKIFzxOcj2NxJ+dNqpiLtreoN8XsfLvfM/46NeNaynG17";
            sout << "jIqcV8TL7MVqJ74/OYv6Mek/0vjLysUDQwPniEkka91ZCJboXzsuW41M5QkSfU/nxsD05TvQVbFj";
            sout << "GBNApTN8Oe8miBahONkbfBB2ks536yBwG8RXWGSHR5Jdl6yhUIw4mWb0WupVEd90sZmXEYeP1q4K";
            sout << "MvdHRuEQbNZfyXLskOtwKsFAqa9ebN6yVdM2tIOOp12Km3mabcUct0nOq///UyfEclOfrC5km0on";
            sout << "WWXFk++6vwChLbG+5WCZl5SNeuxEWu1odbh2XZ5ng+XSz/T2P2l3Fa7ujEKilHAYYbS3foWHO+ev";
            sout << "sZlG35o8e5wuKiqnf7gGs8tAIFFyONtfRZuGzH1qy5W7BysZaohtU3ihEzJdomSI1xKHE7+Dn94W";
            sout << "ZDi7S+Qp6VKbepW6JcdWctmtXyncC5AW+UzquyYuJzjoDRdVe7uEAA44Fhc7nHFHx7snTuZpaYcE";
            sout << "OfIBgzr61aCpVMKX7YJ6NOzRNgfTBIMl24JybjFG9Mcbk2qQVzKQ+w1StjfGTEexWIfYRDxhoUzB";
            sout << "I+2cAA==";

            // Put the data into the istream sin
            sin.str(sout.str());
            sout.str("");

            // Decode the base64 text into its compressed binary form
            base64_coder.decode(sin, sout);
            sin.clear();
            sin.str(sout.str());
            sout.str("");

            // Decompress the data into its original form
            compressor.decompress(sin, sout);

            // Return the decoded and decompressed data
            return sout.str();
        }

        // ------------------------------------------------------------------------------------

        // This function returns the contents of the file 'frame_000103.bmp'
        static const std::string get_decoded_string_frame_000103()
        {
            dlib::base64 base64_coder;
            dlib::compress_stream::kernel_1ea compressor;
            std::ostringstream sout;
            std::istringstream sin;

            // The base64 encoded data from the file '..\..\examples\video_frames\frame_000103.bmp' we want to decode and return.
            sout << "Qld+lmlZhXVX5NDRWIFG1T4+OGdJbkmxAXXdHZEtpDf6knVTlRWyAhgv85Tf11KJZbhKv1dcsKtQ";
            sout << "fX3y/9RxoGwtXxwFXCxbBrLRaklZ0wCJRoLkFEbV5T2V0aJZQJTKWkAqP6JwJ7zPTyCiJkDRbGA9";
            sout << "1cIxGeLInzTUAlhbGO8eexmWzXHz6KhgZ87K5c4n+YcRjVfwTo3Bl2AB4QPv/bs5da4g6jZd7iwv";
            sout << "PYnF6pRCiwuiCobwvi8eG43pWnNEX2bIUtfMVGP2zA3e9Qg7Q7w6/KNILIYSXeI4KjLlnmC9YPB2";
            sout << "2qPH0Q7DVlBLBBUHO8lZaFg1gR2Jb+4rgSvuUg6JaoUbqAUMaFFnQCjw9bwdN4bJ1GFXCu4G2afO";
            sout << "/izIyq+QxXHIy/Ez1TZWTtkJfWRkKAjj59IBD7fRSqyvL2tSatSiYQffAcsyQJzcPMTHahKq3XeO";
            sout << "pFu5XOr3OsyxVr7ME4/GAOcUjNpdW5pAMPNddRfL6/Jy6S9ICHxppdwhMAtbxsaHiKv2xgcc/t3I";
            sout << "J0t4NPThn/QTNed+F4iFR30+lwrS1EyOUiqRT7q51LFDQE+/ZyqCrq8vEGp+BRAxMEoL0ekA2B9W";
            sout << "vceyzfAlymEivr89KSSC9L+VZjE8qylGshgeXM30qladBEU4CKtcTlfJFWSHZX9Gm8qDXmp/5bX1";
            sout << "lT9/GNd2E1hBWWQtmQ50DMlInp8oFsoURaMgzNu8zOelG1WAD663IxutR2wJY6ILYQfXNND70XrM";
            sout << "O2PI0ZVRWenZjLzXtSefvwSXCG1G2u5sjq45P4Uk35wtetzV+N2WzVinM3yYtcABa51N3Se2nDWP";
            sout << "huxbCEVYT0c5sfaNCy7XRQ0ijcrMdqMDbecOuTMSlE8FEu12Z0cNFQqao0R8q3601D39vmJoxIPM";
            sout << "wRIiHH6qVcCMBV3oVPWIiD8T96pUZoN3V9BpyQx43MEC6ZdVFqUVQtiYu/PgWqK2pwk4Lh4vpou3";
            sout << "Jad9ENAsLdoVKZrtHlD8k70MVFOXNXJELqJarxwDy35JAha1+dHBvcgU3q/8pOnhS6xZ6judsmXX";
            sout << "LVBdUnKu4w/tbXgJA7lkLtwtODiXCL6edRCSkuXmTRLqXhrzUYzOVO3lZQSNWLLagHa73ce5B6Wu";
            sout << "rVTjMEVaNC3Guz5/cOkZrveG7MTk/eKR56x7MhkDS3H9pbNiCoqO7Ub52x7j/aicE48SYYkXFTrM";
            sout << "ETpxS8VFa1i5VpaiDxhfZKKV03S/fA0QoUcakVISiSN3VgSrhYsu9KvVjD3kjW9IMKGMlcVH040t";
            sout << "1cs0rHtLRVh/FjpMZlQxWxhI9ZMH0Ccfzpr7SbTN0Y4WfKaO1O/JfqYkzCnFTThvkrUMAeA1+zHJ";
            sout << "1HS44rFnT06RSAGFX6KaqrRFBOC3V3ljXEHyyrGJJe0OHWTUAwov+JzoXuB8onm10lcxUpV2Wd5R";
            sout << "RZ3dx/u9pEYgBolzQIXVkcE0YBYmkXMJfgXK3Rp6V+1pIR9bTWUU76/0CgMHqVBKZOtpl1ykTQ90";
            sout << "h/n2Mcjhmo9BtRIi+R7rBhr2a+oNj7GZHVPaaItTcdSvns42OxGjc8R6ZdzLHzNNPYx0NkbVeKnU";
            sout << "FSV4qd+iYq+N5hiPeTRfwdLU1wRL84Gvm+JfR3Y6eK8dBmMGCp0skspHGgpf8xTXXpYmOuZ1FmXV";
            sout << "YMKmrYMuTm0IigxtpmCmHPmRZ8U9xbe4Oc8UDxkkmCUvcuvMLS9SJKFOQiUJwxuXgf2IzzzePLcx";
            sout << "ePs3jrlOoD5GzQzSvDrJRIeSxjwTrVFiFT+qwMIIAR7TyTjBOWV51+kM0+5mU9g89OwLmO4yxXes";
            sout << "F++dusUFHv+DGSCxk2iGNwiNDQA8PktmdVKux3PpVBxWZ6P0W4Kyso6RNIXk9iAVZj59zCr9uPwK";
            sout << "MzIZ1ASN4a7lrfvv4cD+Uxovvt1s8Kck2wmIiWg95RiLdnopqSg8qZY/3U5u6XmXCqnEFTYuKKVY";
            sout << "dWsDem2GxTTNM3DtxFN9fYbHN6MdbuyiVRPovBz/T+9AYpx0PVaFst9WVLbf8Ph2Hs1S1vgrW5sS";
            sout << "qM3lBgjEYl4CvDzL5HbmqBV1gTOZrzkrEdr00ChKzRURvTBn7NlvPZewMhL+Hc0SVXYtzCLSl+p4";
            sout << "C3tIOZaI2l8JIt8k3znBFs3UrWx1WGCv3GY8HEsTa6uX984B5dVvdlUqmstnqK0FgMEkYxiPWoKV";
            sout << "eYlx3k/sV8GChiZWY/0k15KIY5c1ZUxBQezRc6a8IwYeuCK4dcfkMtn6AUpWu3W/vY/sx4dwNNaC";
            sout << "eWhKda7lBCMsfRdeOZvWgvFKoIknjnkh27uEJu0AskTF+xhdXdKwIfM2ThYFipLGAImNXGuHbtlN";
            sout << "CJ4FBcpn1BFWEY6E4Q+UeOlU1rv//LgKYxgNC2GX64oLOAoPGgEOG6dTKmnad/7y42k+kmQnvLvT";
            sout << "YNkffPIcqqBvhZCmvIg4eMR+cB5Kw5IN1tik/XNmrI2FP8y+7OhNlFah1F+slgX9oHVKoRLtJFPT";
            sout << "e4rQDWqAnWMAoSqsITbHCq9Em1lP6Cq3anR7zsCNH79KJCsbZOZ+J9ccoLm7uLJQUl9BW4irlZH8";
            sout << "fYZypsN59YE3kzHivG1xH0QsDLXrw4G9E2Z6CFojySfluCPiHXO9V9XxiMaU2X4uYHVb7ehs+/TX";
            sout << "h8uPIngi2rIspfThIzUPC8nl9k8xIH1BoWhryXHkzPi4M+ZAI9afBNr6dB9qa8O5gPvcfpeWPRvB";
            sout << "TGAw+OW3GX40CxJhqQ6n1G101LjjbFvWWdHTm94cR3cS4B2BYzfAqy9Fe2woNh8OT9520VCEMgWc";
            sout << "4xYjf8rbQFq3kBgPDVEyu86VZCdVBaAs91vTnfd1i7tKCN8JECyr04O4qJ5IM1+5Mtv1Z6wlgT4r";
            sout << "NmW5FaCZIKwfORsXG55kuEeKBmvbnBUd6iO+jVtIn3D3RjWCHS05cBsw5uCvixNIHXIF48lTiHpR";
            sout << "hGITp+vcydTLUEeVRxnlKIn0RJX0Ht2PWFNNbU0RV2WFS4K8QGQxekj6i3XFsB0UWJn7ipi4eHJl";
            sout << "CIP6hG5bQ0qG1tvhC3a+Ve4et41gI/DZu9LbSqg3oNWUPjEgFSFadjJ6ZByhJ3QTm8Pl0uVdpzra";
            sout << "myKEafoz0tbaRgJgqAGpQwkIPgp1ocPZBplMSbI9zjBhrTlI6fXfH8aOw4FgB6F436yGuVl3/Rjn";
            sout << "UuvOT27jH6TvUOGxWBbzXXcxU6250E9S45QSh17Ly1/pfal5GdAcemaHKLRBeCLMwMszIr5ZOM8M";
            sout << "LxeGCXvWuMuoLIPmElzolFuKKEht5zAr+KGbhsILlLcVpQBnQRM3NxXfh6bO86V/lEOZxhy1b+Js";
            sout << "1bQrn1Uj90QnhqDoI8t1P6aSFDWKvbsnzC30gPZ8E+FFTwsvvKlzijcoroFQ6I5rSdVehv8/YNvf";
            sout << "1yYpXA+hIK5wbbyIAXxxmi+5OP2viwGEihe5rwkuHRTmtJJ09rHWFT2TbVQAF7775ZFNpSogY57b";
            sout << "GJVCa9pBGN8qzsahk99Z3tjeP+fP+4F4Si8JyPJq+xmTO62ciKxHrHt9b6sE9H3N62huURtpP90g";
            sout << "91QbEbFJU/5JSPfJjJZIp299UhafpBSUACpOnnjBJT4xhVU1E3k5x6TKdrZYzAC/dsQ7BEwZgg+W";
            sout << "5wfLTnkJsuvMLQgS5e+Ot/EUNd5+16kkR/wFeH6vtzG0EyxkWK4V7GrKtsrLWV7jOC6yY2mGwfnB";
            sout << "iWLdaI9Wj9pIeTk7w+PcyTZZfdDldyOF9TwYKvHAz6ea5s7sUe569Y+iDqOOXOwpjX7TMUVgyTCW";
            sout << "4VC8ixQp+UNTdlKLoeS+2fw2QupH395hLQ1DHBcO5Tuil8Yqxxw6nV+j27THym0CaB3j0JgA/4Tq";
            sout << "vbqsEonwHgSaP55u/j+0QRpycKoXfEkS4gh/qMhbLAPQyTNY7QLjE5cVUqzLG1twetMMcXZL2Dqf";
            sout << "dP2uDCswsX+RJdIAN8BFoa+J+uXHgBMgbxO4DrGz8GxuSr8RfS0JeEFReLiLpE3n+SrnLwKAUr8y";
            sout << "qWjNrqRD6XKVWpX4SY1I6wcE9/3jX0c27mGX+87XnUZdIOZbNgXmntyS8J1P4uFaEZIy2rC2ahgg";
            sout << "1mdwALsK29JJ5QEaSg/qd9UEmm6jnWLfquCgBIkacJgjG1wc/029kL0/Br9t7tuI4jPiKKp+XRM8";
            sout << "ZZJGwjcB0DbJPB25JQ7zp01lbnEnNiGpNkNXUQtf7a9Z/F530W0M6wG4fOl3lW1HP2zF1Ad0GJxK";
            sout << "y9g9bj+tCsfJQfeymW/5xaC1C+9sEDOsg6hgVH6avh1a0ILiY/5MddtCHDQKQF5tgkuiKXG1+ihQ";
            sout << "CUlFov0Q7HG9UruG+eydcYSdyfKkdTqYQ6F8PjLP/rpiypX1jiZAodzEuvZwQcZLAgp3gqV1swZw";
            sout << "6rm2JpyS2Kqc4yCanywpHAU8LkVsjbPTanJiOgDguguVRtYGpAh4fTje2Kofw3rbPjznNu+CC6+4";
            sout << "8QxPZtcc8USR2EShYVlE90d5uu13F+epClENFc5hww6ymXysLDCIop2vM0PuA6csUxJqf6YFZHBm";
            sout << "c1h+7m2ULTr7DEVeRCbDkLvrXrL3zzqESvN2x7QDUW1lQRG+A+hLKS/nuibs6tI8uCZ0I3/VDQKX";
            sout << "hAhNjD9x82FqKKyPGFikNASTaFg7I1DAHm2iHm1Rbff4I5S57UUIk5zr4o/pfQBh12XxfpLtvSfn";
            sout << "SGXRAUcTO5jYoftuSmX9tiMfjjSXopedLWjMKSUXE6m2yggIzxfAzd60Zyr9lNMBVzeXTSkrReAU";
            sout << "QjMxJ9zsAF/86q2lsxr4gJKNmg9QkunRbsVlh3OXE7MkxBHlb2MkULioU83Gf8LiosiEEmYSB5vY";
            sout << "H8D6cgnXtB64BWoEJpQuG/R+zvnrWNHKmcBd3YkEi1j0dtL5G50SlMOJuai2xsD4RosbPEdV6Qx7";
            sout << "ESWrAO1naNHp0fewaVl3d8qXh2rFZAKNZXP3O3H7h3BAoMOsxWMFNJwQ0xZshhRvASlOr3Yn3HB4";
            sout << "fOqqLiPecX/o5r5FRa6qIfP75QaKqcu+iA23psV4/bszP8L4wREKioaSFncsUwsMU5NyZn3hPDUo";
            sout << "kdW+ZGGQk1F6Md46VYMIvSxC0xer/NcISrDfWmae5WCoGlJGhv1rSnSjRmMrj213eEh2qpDNhz2C";
            sout << "8dlByOZtSTOPejh42nYMZKCdBTK1lftdSeYdsHRGt51TQOauYYNjCwQkh87aW32Z5dsUi6H96k/c";
            sout << "NTnEoTZbmyqFdmov0QyQBgIfPmIUV/z+3x9PnfuVHJr2PBdElqqgp7h0Z+Z/7ESnDV+G/0/dx6nf";
            sout << "hfktS6/8axrd2xTKr7VrfBLvasV0dTC/GibZlQ1hvowffGgH8dgooP+Nc/S2UqhkYuMZz/Df1r5g";
            sout << "jYMY3rLQt3RqpbZmwU3p9MGe74SVOSCLgYXIDQiGyoGGPuwRwj4uhI5f26fC0UzG/0RCe2g4dDe1";
            sout << "vpDGatzd+3ixmyzEf6K+4UU4l7V7XAuUYs3lGEmtkz9li9mF3huDAUgOvvw+jErQrMnUGeyfzqDF";
            sout << "xyGfB50sK3Y1q/NSxYKQ1oHsyNYvnTr/aD5OPgMzbjv1Ii1mnwtP3m/EWmdSmAsvsjEq6Dy/yBuv";
            sout << "WZ9JO/XtxbusYXBNUxb6lU00DpJK5NGZVxv6UiQmrRqZvg531ratwkkaH0ZJSOwOzWlRAeVKiDPM";
            sout << "mxtnqoUGNcyr1SqIHBHvhFpr2BJKEeZ6MRwNHoQYCA1dQ4l2YJtZVITuL/o0SHNq1EXC+5sWB00T";
            sout << "UFfHRWFOYH6/baG2GJSvgzQrxBH9miEG4WzXQcdTQlmup0RZWjnwEoyHRMj4pc9cujSsTMaGktXD";
            sout << "ZzkAX52IncX0jqq2ZE4epQnx1UHg5IXANgb0Ed0y/FQjS24SZ57uIvl396ubCO1LQ04vGH2Mvs/m";
            sout << "bSM7jKGtcvjQ4uvVaCyk29Ek4EKJM+gUAuEj3IcKnoRNL/P7L2NEBlJuQBlIEgczCVG68kkIOFLZ";
            sout << "CqGgbf9IaBULEjthq4Pr+B5bUYmoBGtkUDM0KewVRJ+YA6A62AAAeStbx+myMgTTqq2b4idEllZn";
            sout << "0Hc4kztLuBgtHfwaizv9gYEUbgKbXQWfTQdUNAGSyKyeY5dQ425q+eyHezNiXZ7C+tDiM2UN+Lg9";
            sout << "t49U+7urUylh189bDcZcGmbx10unLEwYGT0CgDPGA8DJRlkHUkXSlImHpz8+3hjRtgljVcz+kFXi";
            sout << "jP8F2VcEoiUDRpaTwdJTi4pRsWSZF6pDEvHWntpnhtBI51zoDzEpbkHJbjRbvBA2zl0b2MqmwGXW";
            sout << "2qDOXOfSFVUVzTr7++omJ+UXxq9awcr/LVXz2tsuMyIvREj0Fx3c9jVozDSbOPzn97QD9LNh8Pau";
            sout << "bvPgfmMGz1Xj5f1UuKvUiCfvfP8ZZs5l08ChMadB64ipgmdWKK4adqE0ES0cTqcg+pJDDL8FgLYV";
            sout << "susPETHqp58vc13TMcBaqMAa44/xA98rFy+KZpYNxFf+l2U1eq44OH/zdytXhOg6y8TWrKYIdX1A";
            sout << "C91FZWXj0t5KfTahGZE5t0nP5iKl1IdXnE+dOIvIHuzrTnyjM+IiHmj1DxClG1VQXhZcwvfBzIY6";
            sout << "MRwFxUJXmt7O55uesEbZ/anmQB6LHdy7hMuvvCgeLBhoRnnyiRJYp3OAwNH5MLy7LXK84ZKTkodw";
            sout << "BuyRw5oJ9KOB13Buxj2yCRhLky572BT61cw0rXXkw2ZOGQgzHbO+cjikNk5LsVyMYfE49ClKM4w9";
            sout << "6pVtXDNzSPvQK0KSO0ZMKPDAr1/MrV0PD5Y2v1JIVPOuKSjq1s2vP7ypzarrFHAJnsD1FrigYXME";
            sout << "9amdaYhCmPIiOjj+9xSfmuBrrsHJ/THe82pg0fO3mpsF7TVY3zQCpiMDY/UqDyRrgc8NQu6A2WRc";
            sout << "oGCRQ5QCWCTFhbIV1HQ/KXjiCeFrCd88KkZDLDwrWMqeeGDTM8eWQcTIqoMMtu/JiCrts9c8yiTj";
            sout << "IFiHTXnpZMYwxDeT9hvNkifueXVjFYbEs6hHySwX0kcUSgtRP3fUKyQe+Iz+u/iOi1tFHQgPsV5W";
            sout << "gltpMLE72aQmlQ9C1lYkJQoKrQJRFiJmiQn5S2hQKXurP2HOIsAYSpW4mpeeFJpxsF8lekzeniil";
            sout << "96nrrpQlpc1vPdDhbmTY07tGoNlYRjoKTYaelkktnWA2a4n69x+kD8iBfVLANU+EFHNfCyQgmdJe";
            sout << "fWMYQwqwgxMWk88peAftdx7bDJ1ZJQ4q0zzkot7w7NB4oe0Q66awUwj2n4DYvDAum3SRz2LwsaiR";
            sout << "Ke8N8dMtRYN2mosZ/108MktIKlnJ08bkI6eDnO5vHlbeBfLOfEP++jGsgdS1KHrZHdIygtioQ9Zj";
            sout << "RzWb1fKOpoIbxV4mgh5MwpliqwHmG0DIq1UPIEqj8W/Vbd1T49FhWRJs8cyYraimx9MPklhyxlGA";
            sout << "HvHpQfuihMzLT7nNAVceWz6GzYfHy5uvje2MYBVHkoQ6RaMnJUCfFd64pbFTmsB+/+Teoo6q4I5G";
            sout << "fbeIMh26cs37X5MwocXcq7jSunVLgPJYDlBcbpIdhDt+Uuf5VPo1GXRAk80ouu1E9mrYvlZ4sanS";
            sout << "5EJ61PNevEOSDPPdYWwfMR0rKZuoefUCNfvy7XyXpEnZpNLHKsO7Cs6ZyL6zQykBk+KQCfdRnTHa";
            sout << "SVSpKvmD34O1gGkrq7kSfyWO6eU0TU8HImJY1tpI8fcMTheZ24hxx2QsV6IbsM0wbxS3d2Qo15XC";
            sout << "oP9xQxVcn17iiHUg70awt7j2VFEuCb1QjMbNnd5lUvjSJ0jH6gi+n/on65AWEJYm73Akn9i3fP+v";
            sout << "oLTDxIOmjRZsDgzsAbgtxb0Ozu0rNutnPRH0Am9C+GGJX78X3u5qDVn0q2+qD7eQkPOmW6oQiIYq";
            sout << "8rWM8ya2N940j+H5HsvyW0FKyrqjSSXQ5fF2LJE27Wc5ob0tVuvWAzOQ8COZWMMapu/S2C61f+2s";
            sout << "dFiCjJQqvr3Ai3pePmVrODBGROVHow3PXekC8iQLRwOYfNHwiM1jGwPOiMBZ3v+LghoBYHhmUh0Y";
            sout << "kZUb1j5Mncmgz3pctYS7KFhlHq1RHTDlF0pevtVLDXyy0QNXFoOqGydJaqLN2eILSl/eOGmsVlQz";
            sout << "lgyoCzPW0YsuN7Z4m0z0CfdP7WTpL4s2jlB7QpdaF3JOaip2nVZzX9xb6qXPosQ1WF9yixcznoAZ";
            sout << "bfWnipe5Lnxm+KHX8OUclbWUC6Twaxoq9Uo9jXuwkU6PZ5VzM5lao7D0JCPfVo7b55q1258j1AnN";
            sout << "9JNYYcZ4ZjTEKvbXGdtb943Wmmr7CnI4IyKv7AFpeiMAYPVRgxIvn0RjWpJ6lzV7LutAjv0HXwZ7";
            sout << "5XHtqvvuIe+IIIG7Iqj+JBW/pgQsbb7CKXiRmrWwl1MTtDIrQuh6UP6mstikBOF/d+yOSqCvU71B";
            sout << "A/Bj8gEqiPEqB7uih1Kakz9wRraRAYo6VZphC8Q5LdkFQG45Bj7KnACHbBs7e9xe7GDrP4OVbQHh";
            sout << "m5QCONkNpd+0umv74Adr64LDpsMr6CLBgHRrXukUXUGmFPqUCD7lC+quhu8psnZZrvSNPSa+goRp";
            sout << "PnO4wDlSeBSkuSdyC7utuC40jQtIZrkJmvkq5UURj4VcltYQ+5KiypO7ixAK4fvKyBTgxV2WNVeY";
            sout << "QvxNeJcNck+IPEhYLyplcLzyoekhnP4ZDD6ALG2jWZ8QoKpShA8+HdRSJoFguoERQRS7ePlMxywb";
            sout << "kcLPkcyEOlBu7lSjCRvjvdROmLAidz2GrlWvkuLrRe6clvs+U8JfBTjOON6P5nq41ElWNA33r9KN";
            sout << "kcCmEICyFl/Ier/YdjOUtXaHsKwHaMjmrXHQOTP7ELs7jouCbb4cTFVtgB20smKj3paJgoQysxmq";
            sout << "12Ur3QnHaHjaoxzIQnlaD69TF4O7T6U68/C6BwQDuZiD6iB+mkCQ7Uz0zFsiVMbF+RgFJKpjN4oo";
            sout << "SuR2gUL8sPlixeYmannuPwhCkJ1swN7NiVeUwTGFUDeQt2eJvJc2CllOpUZwF9PbyHSkxkQLs9kP";
            sout << "9DkvDhKkqJ2aEHDlPGBjETFYjNGdoBfMIcr1yvWshl9fTGlEi8c7mxMW/5Ed4sSdV3N6qRj3KZB7";
            sout << "zdQJ4Ql6tH7zkYWfo/ZtTEggQQAe6FzyQ12/xgFgdAulwF1IJZ/JlzY/SKePQPX0/89AooSUAl2q";
            sout << "/BfEFEI5XJhXDp2VJevsfFAjlmf0pbC5DkyBlUwdStZAoZpvfZPXcQ1NshCuD/hT4FinYmLl7GaW";
            sout << "LJSXkHegN09TBLWDdWe3DfKDZEWYqbIvtgt5Evgtw/5Qvy5DtefyHPq21BoIT8C1zxkgfOqEhH+l";
            sout << "E/fLaDsROj+RUJ055ycF9Wa8mzCOZMnaqsD13eTXtEY7LQ7MEo/15S1ny5JxaGNHxy7YJ3JfrshB";
            sout << "0pZ5vL3jvb9iYwocPnlmFpli+Md6qEdonEN6K86WwcRNqV57mMlF/EhBZXi8VnW0nqglR3mT2xTz";
            sout << "CKiwPK1W/zOnR2K0SsQrHV5kaLCfRi4vM3Jv/bGjlgJSUYBkB1QC7jBKE+uA3H5EmspwnDKPAb+b";
            sout << "kYXzpMWfQbvMvzpumEs+nY/xbCf2Hr1vhZAAFR1L4F49xrxDr8rjP3DryRgtBquQH6/5qMwdg9pR";
            sout << "4ACieRQts0S8jC77fxLAFW2GqifC5DBhwIcdhOhxR1FGLo5RzNktZkPob/fXEcC6u/Uz4v4Gy0j/";
            sout << "ZM783wye39lB2eymWQ5XPGC9FKR+ZLodyKJK+NXDBiFBXOjY4WNKEghllE7jAEU6VVmAN8NDkQdf";
            sout << "u6MiVlBZLK1cepTPDj6G/TGH2FK0I4O/z+Gb2WQdZ2VUObGhf3GpYNR4XEoBsJe0I8F+TKpHYpdY";
            sout << "guyaXCEMW9XpK0hTOBXfaooIQu4+Yb9e18fKnkrW74Aj45mdseL80RH2RJGm8LeZa/Nluqm71cqu";
            sout << "9zJkwZjxlfICDgLhkOXtt/Lihzoqav/abehnuZjlyWwJ34eIhCmEv9D7C0P9VE/7B4P1S/demat+";
            sout << "W+rmgFScowUvs1lfxNr1I5uCyLEJ9VphPUXnykyJ0XM6zm2Za8jD17Nh82k/eCX+JFMWv5znDNQe";
            sout << "aDzkIRjTQhXWt4e6GNZ5zU6g9rr/TIePeMJyRE4D09xOF6orHWMBh/TFAL6PQnJaI6RUqCg8le0+";
            sout << "ySd9qWCRpZsYBK+bZBEV9A1iJyvhLyKLgSGC8Q5Y0tQmE5zWSlYkRr4YtfASRn7e4l+Hz+hhcReD";
            sout << "F6s8jWCAHjsq4uZi8qf17LwZu2xpbSLmgCE1sij59Mli4sWlPDkX86ehGqvH2D2x3hKlXco1v+q5";
            sout << "dbNe/zvwcng5KoCwL0pmT3itlXfQs08+1lXbpoqUf2wqmQFEi3/TIWq+1lF/zV+6ICVx1d1eKarh";
            sout << "ZgdmD0C3QLlPNlUAG1j6qWQ4K32ICyKu2XWRjcaElm8uRxPBBJ2DKWJFyDwP8caoB80A2ApnnFCH";
            sout << "mrijgRO7LeQ6bQJJjOBSjeppPm4jTsVd5tlAefLV25uRwx4Reif0e4x+24fxtPj9udHRWggwGu/1";
            sout << "Zs4+wsJ7KmX2ekZ2LR5aotwtBCo8X1J9GaHal4WCzh9G7EqAeSjjrD63DoSgJx8PAUoH32QXUMxw";
            sout << "b6JR8Czkimh47Uv+aIqZsr6GDskW7xfxxgmzIPwSA0cJkulJKiOYEmjcfgzqaKHrcjMRb4LhJRTE";
            sout << "F7Xgdk4nJGT2hRv3Abqz4725TLPUsfd1DFTM8BJuJSWjGNa2cyQ6f3CTQPEZYFiyYMCtd4Ycytnq";
            sout << "N4djAycInnDJV/+XSjC94UB04UDvkq5lds+tgGMDf5YWClAz5EO0ztmL5PIQrA7qx1ykF7xiRHVB";
            sout << "Jy1Ln0oTEV9yf8jlIYILg3V/j716p61dkyJIvRtO6XhFdyE+4ia0WdXaNWNwvx4wkzqRHD4IxMD0";
            sout << "imBJe7IIMVztDzJ7T4K61P7nzwgGJgXHwfdh4xxYZC+4Qf7LVyGOeH30AgAiZBzEFu6kCHmPMxgZ";
            sout << "VQIAvbW0/84nBgbqTvUTQTVytf9ufnhuVZSJfXbMlFGTxeCtHG9C6rgAAk2QNPHb8up8NL4ZnVcA";
            sout << "AWk0OHyU1DZrLFb26nmHBxwVZySrSppYbldjEYxMdewD2OIYBYrCBCKIJzZpTMP8D7QOIX7B05Vq";
            sout << "zPH2i9cAAAWjZjAkQxG25u2KhBSufKHwxVZ9xgi82dPIUZJRMhNOyUQ3uVhK4MilNqxi3Giw2KtP";
            sout << "lO1IWReKc0m1pXxv70r03E0i7kf19iQ6AAAABRagQbu/yzAW7Stqw18BH3pVGZSoj8no2Sn/A9QF";
            sout << "E49JB+qp6BejcC0vGwH+ogKo7pk9fecrkHEc/6aH1Dt7ciMaDXfFII5rE9AAAAAB5kmFhDYODCNn";
            sout << "qZy5bb542mHiW4BxDLW3mQ4uKyXFV56IuntOj0rsjRZdrOATffLebKWyn1anGbGdQvJcOZcHvV6h";
            sout << "1Aih5pmFfPxyhA2ba90hh+kPYjhT2QAAAIYcLHepOiMlf+gULTiFNjbeyaz5vgWcGTiwkv4q4ED3";
            sout << "ZLE0lb+l/7qt620m8kgPvUAOLnK6ysWdLpoTtevH7MY5LafF3R/YxEPOo9xgr7zDLrMKLkZILeIb";
            sout << "yegmBAnGeWonZpv3eidKXsbBFhHGw7DoSakVKAT7Jahlmyk6howH8x8VEVpK1QNbaAXt/pFXGrzr";
            sout << "dAO6LjOcrwxvW8iYPjPChEHqwYloPizhrN7k0u0F/oq/urAZxyJ/MP8Tx9afuHY4aocW8FIGmN8f";
            sout << "NSozaWDqqw209VkydVcnibQweQucq14zNKxGZgvnuw3SKeLPAc40Z3fSWsEfG/DImcOJ6QdmY8S8";
            sout << "34MKBXUjR8W0w1P0gs407lSQFRZBEmiOpU3HW+A4BisSe4fcFlzFt+lhP9VNDr7lBHpbAqkg1Ob4";
            sout << "8/uu5w5qut8RiDJp2dYQFY1OieuY1lEN1BFYl4hoUuBQqGBpq3MDn6V40ZzXHl2Rbl+u2tXhNICL";
            sout << "FPOFo0Y+7b0ppLIQGPAd5fBGTfMnibWFP9Bw8KDo9wUbvGslyV0ny303hYXO4TzIliPUUu+nsJRc";
            sout << "pBCeJdyRvQUjr6BjCXrNVSldiB6oCr7e0AfMBU1BErZAhzseZYzzlIavra3tEOdW4xweFxOO1JNp";
            sout << "BqoG5a9gozEd6VQvhurdttGa5jsSm/tEmGDZl0t14nN18hjVW6KCb0u3+kCbcCRRcFz6tZ3+eomJ";
            sout << "TQIBUJu40uNzrLmur+zJyT7+JRRIq/xXSv2R1oJhIiYGe9P99wNd13TuTi5Oyx1b3hHJdysPjIt1";
            sout << "/I8UGCpIQ/FCIEEplSCkhDgedfL9OtDOnU/+bI2cipB8tWSwpwbczIslv49e3+xIqBzLf/4gBDd2";
            sout << "4ZYwe2lHWGL7OXGYCAQbC9ELg2KuRDpLrG2ad4fhUuAnizCLb+q+3Nd7qgkPXweeEMAbSHhp0vuN";
            sout << "XSbVB+cDaHkw7DxxJR0nn4IhdbkuQ54G975t+Wo5uPunpyZ5QvDXijG8NEhIAytpMmhXjkiyUbk/";
            sout << "XmaEnaSTOxOEVh/fBjeGfxHmjn0TgkQlw7GLHa1rB6Lp3y7hSWoZW0RSn0TmVQbWBFHen0cfhsp+";
            sout << "zc7gTzMeFxhMf3ZmBS9pmigcVHPAmU+rOLTtAzaklKumgxf0B+Su8YLZpcvc6Jy8t6rJgCYwIhR+";
            sout << "BiKEN1l6CxFeQtJp+2Do6oEZ4Lfk0DoAnJdV8l5DhkiXiuwX4zPFiKZQtKSBsrp7bq6dqjDphbgL";
            sout << "1A6JahmxkTYcJVOnVWG6LvfHYCIJ6hwDV4J23P+tNZtLCzfHPnDDdyJCmBExsOV1xXzNeGhqZhb3";
            sout << "kSWXntkMAstuTSdgb8vPGYSyEMsTx+pTeXFGv3TN+YCdQTQIovV2f0KTZMxmjoTmlZKueVeFwzSP";
            sout << "S+G65EMEcgQUGloWsFPdIaRzPLYHUktOcJc4bqEDvwIUDLqfjH4zr7kgF6i0KfRUMP4Jrje78S9G";
            sout << "C7o9XJLGWbS32yrdKVlIG+9f+2YQRrZECzhex6WjV69/JSe0jbilwXwsJSKtY6jFik8eE6RCZqN9";
            sout << "c1vOxMcd+5twMMkmnu4yvBEV6Y207UtegGSHaESpEW1MTvbDk/75X46WMSECbeEe8lzlmNaaWyWY";
            sout << "C+siHamjyTgHAvJLEwiNlQ26LS3mJr4HmpOVizbTykoe2WGV/jmUrhhUg2AN0CrS2LqriZolW/JX";
            sout << "jJeehNlJ9Jy4e4NZE2Xn/Q5UC3zVJm3bpbC3JKre7sALksQSQInLf7OoYArfwgYmAdMWhgxRakIv";
            sout << "0M0acqOuNUxmW0k4YEzEncfjSAQ3cAUkaL1XS55I5vm3o1RPiKBjjypzdi8GI7migOK5dRrIctP4";
            sout << "U0S4L6uojFipn9VGhIjGiv0Rg6OcaZO2FDq9SOGmLZ4b92sE7v4rtEt8oxlbbxPdgT6CLQkVCCJh";
            sout << "uvjCIeAmV0wwhaKYzE2TpSkPFrtCPWhBjSulx5OwEGvdgzUq1NAnHBqlc6Oazt+6Hb/b+zpziei2";
            sout << "huprTtScOqJcYzqJYGIwYebjxpVM0f/05a3LodBUWUNUIRNyzJpyEZluHCupDqfEgF7ObUDvtTDO";
            sout << "Zl1aCRWbztt2W0JE7SSCAZbzY+3Dq+m0VUJj1L7NZkg5py0mUHHnz2lDveZjtHr9UBqETzN2xCuw";
            sout << "8LQqbhDKR5I5ThHilGjZjxSN4VUKgBiBNUHXu8D2VN9RHr9xjQzmnSOJPHtcFuf9ioGO43OVtV+I";
            sout << "DvE3YjEt53F37uZBmvTQD3zcBvgRoNF24j6jjTeHE3I/MROKotCIBkxk5AZTF9Awzha4XwP3xOAI";
            sout << "FyhCle92a3102sdU/azu+1n3Jq0ZeifCSicBjrAHgQpVM7afn/yha/YKXqiwingX4pvrN6IRKADv";
            sout << "edJG6C4OttSovas7ayKdaTWURKwQWQ2NQF/24DUEmfGkWgk3e1uFjfdBNFez7/omGkRqvSAoT5pe";
            sout << "rXpbWzb7nigtdMRUdVMdzAHooSObROsbHc7ngUn6sJ6rD88EJZH94kk3adWN6plDDTyRe/eu5Ah0";
            sout << "XnBPmYIgVSYSZ/X6BvS5AZDElImmABuvuLEjOQ3Ydqc1NQIGOgMzWhn/Dc94YMxSOdmHCo5RyGjo";
            sout << "Zn0wntZDa+gKaR8PwzsIi+Q7RvReVKP4xKNPbI3W47v3sh1PeidJXt75OoHyKpoxJJcjwB8me4iZ";
            sout << "ctKBz0QZ5ZBuywtn2Lq69I5nmjHQbpIB+zmPqIxEJOworB7Qg6thkda97KcHsRbMJAA=";

            // Put the data into the istream sin
            sin.str(sout.str());
            sout.str("");

            // Decode the base64 text into its compressed binary form
            base64_coder.decode(sin, sout);
            sin.clear();
            sin.str(sout.str());
            sout.str("");

            // Decompress the data into its original form
            compressor.decompress(sin, sout);

            // Return the decoded and decompressed data
            return sout.str();
        }

        // ----------------------------------------------------------------------------------------

    };

    correlation_tracker_tester a;

// ----------------------------------------------------------------------------------------

}


