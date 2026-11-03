// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating how to classify text with the dlib C++ Library.  I'm
    assuming you have already read the svm_ex.cpp and svm_sparse_ex.cpp examples, since
    we are going to train a linear SVM on sparse vectors here as well.  In this example
    program we will build a spam filter, which includes:
        - Turning each message into hashed word and character n-gram counts
        - Weighting those counts with TF-IDF learned on the training messages
        - Choosing the SVM's C by cross-validation
        - Finding which n-grams the SVM weighs most
        - Turning the SVM's output into a probability with Platt scaling
        - Saving the whole model to disk and loading it back

    We will use the SMS Spam Collection, by Tiago A. Almeida and José María Gómez Hidalgo,
    which contains 5,574 English text messages, 747 of them spam.  You can get it from
    https://archive.ics.uci.edu/dataset/228/sms+spam+collection
    Uncompress it and give the path to the SMSSpamCollection file as an argument to this
    program.  Each line of that file is a label, "ham" or "spam", a tab, and the message.

    Text gives very sparse, high dimensional vectors: millions of possible n-grams, a few
    dozen per message.  The classes are then usually close to linearly separable, so a
    kernel adds little, and a linear SVM trains in seconds and scores a message with one
    sum over its nonzero features.
*/

#include <dlib/cmd_line_parser.h>
#include <dlib/statistics.h>
#include <dlib/string.h>
#include <dlib/svm.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>

using namespace std;
using namespace dlib;

// This is the type of our sparse vectors: sorted (index, value) pairs.  Both the n-gram
// hashing functions and the tfidf_transformer use it, and so does the SVM.
using sample_type = std::vector<std::pair<unsigned long, double>>;
using kernel_type = sparse_linear_kernel<sample_type>;

// ----------------------------------------------------------------------------------------

// Reads the SMS Spam Collection: spam messages get the label +1 and ham messages -1.
void load_sms_spam_collection(
    const string& filename,
    std::vector<string>& messages,
    std::vector<double>& labels
)
{
    ifstream fin(filename);
    if (!fin)
        throw runtime_error("Unable to open " + filename);
    messages.clear();
    labels.clear();
    string line;
    while (getline(fin, line))
    {
        const auto tab = line.find('\t');
        if (tab == string::npos)
            continue;
        const string label = line.substr(0, tab);
        if (label != "spam" && label != "ham")
            throw runtime_error("Unexpected label in " + filename + ": " + label);
        labels.push_back(label == "spam" ? +1 : -1);
        messages.push_back(line.substr(tab + 1));
    }
}

// ----------------------------------------------------------------------------------------

// This object turns a message into the sparse vector we give the SVM.  It uses two
// families of features:
//   - Word unigrams and bigrams, e.g. "free" and "call now".
//   - Character n-grams of 3 to 5 bytes within each word, e.g. " fre", "free", "ree ".
//     They still match misspelled or abbreviated words: "free", "freee" and "freebies"
//     share some of them.
//
// hash_word_ngrams() and hash_char_ngrams() hash each n-gram into one of num_buckets
// buckets, so we never build a vocabulary.  Two n-grams can share a bucket, but with
// 2^20 buckets that is rare, and then the SVM sees the sum of both.
//
// Each family gets its own tfidf_transformer, since their document frequencies are not
// comparable, and the char dimensions go after the word dimensions.
class text_features
{
public:
    void train(const std::vector<string>& messages)
    {
        std::vector<std::vector<string>> tokens;
        std::vector<sample_type> word_counts, char_counts;
        for (const auto& message : messages)
        {
            tokens.push_back(tokenize(message));
            auto [w, c] = count_ngrams(tokens.back());
            word_counts.push_back(std::move(w));
            char_counts.push_back(std::move(c));
        }
        words.train(word_counts);
        chars.train(char_counts);

        // The hashes can't be undone, so to tell what a dimension stands for, we record
        // which n-grams of the training messages ended up in it.
        names.assign(dimensions(), {});
        for (const auto& message_tokens : tokens)
        {
            for (const auto& ngram : word_ngrams(message_tokens, min_word_n, max_word_n))
                add_name(words, 0, ngram);
            for (const auto& ngram : char_ngrams(message_tokens, min_char_n, max_char_n))
                add_name(chars, words.vocabulary_size(), ngram);
        }
    }

    // Returns the n-grams of the training messages behind dimension j of operator()'s
    // output.  There is more than one when their hashes went to the same bucket.
    const std::set<string>& describe(const unsigned long j) const
    {
        return names.at(j);
    }

    sample_type operator()(const string& message) const
    {
        const auto [w, c] = count_ngrams(tokenize(message));
        sample_type x = words(w);
        for (const auto& [dimension, value] : chars(c))
            x.emplace_back(words.vocabulary_size() + dimension, value);
        return x;
    }

    unsigned long dimensions() const
    {
        return words.vocabulary_size() + chars.vocabulary_size();
    }

    // This saves the learned state, i.e. the vocabularies and idf weights.  The tokenizer,
    // the n-gram ranges and num_buckets are part of the code, so a saved object only makes
    // sense to a program built with the same settings.
    friend void serialize(const text_features& item, std::ostream& out)
    {
        serialize(item.words, out);
        serialize(item.chars, out);
        serialize(item.names, out);
    }

    friend void deserialize(text_features& item, std::istream& in)
    {
        deserialize(item.words, in);
        deserialize(item.chars, in);
        deserialize(item.names, in);
    }

    friend std::ostream& operator<<(std::ostream& out, const text_features& item)
    {
        out << "text_features details:\n";
        out << "  hash buckets:  " << num_buckets << "\n";
        out << "  word n-grams:  " << min_word_n << " to " << max_word_n << " words, "
            << item.words.vocabulary_size() << " kept\n";
        out << "  char n-grams:  " << min_char_n << " to " << max_char_n << " bytes, "
            << item.chars.vocabulary_size() << " kept\n";
        out << "  min doc freq:  " << item.words.min_document_frequency() << "\n";
        out << "  sublinear tf:  " << (item.words.uses_sublinear_tf() ? "yes" : "no") << "\n";
        out << "  dimensions:    " << item.dimensions() << "\n";
        return out;
    }

private:
    // Adds ngram to the names of its dimension, if tfidf kept its bucket.  The bucket is
    // the one hash_word_ngrams() and hash_char_ngrams() count it in, and since
    // vocabulary() lists the kept buckets in increasing order, its position there is the
    // dimension, counted from offset.
    void add_name(
        const tfidf_transformer<sample_type>& tfidf,
        const unsigned long offset,
        const string& ngram
    )
    {
        const unsigned long b = murmur_hash3(ngram.data(), ngram.size(), 0) % num_buckets;
        const auto& vocabulary = tfidf.vocabulary();
        const auto it = std::lower_bound(vocabulary.begin(), vocabulary.end(), b);
        if (it != vocabulary.end() && *it == b)
            names[offset + (it - vocabulary.begin())].insert(ngram);
    }

    static constexpr unsigned long num_buckets = 1 << 20;
    static constexpr unsigned long min_word_n = 1, max_word_n = 2;
    static constexpr unsigned long min_char_n = 3, max_char_n = 5;

    static tfidf_transformer<sample_type> make_tfidf()
    {
        tfidf_transformer<sample_type> tfidf;
        // Ignore n-grams that appear in a single training message: they can't help with
        // messages the model hasn't seen, and they make the vectors longer.
        tfidf.set_min_document_frequency(2);
        // A word repeated 10 times in a message is not 10 times more telling than a word
        // that appears once, so we weigh a count c as 1 + log(c).
        tfidf.use_sublinear_tf(true);
        return tfidf;
    }

    // The words of a message are the pieces between whitespace and ASCII punctuation.
    // Bytes of multi-byte UTF-8 characters are not delimiters, so accented words stay
    // whole.
    static std::vector<string> tokenize(const string& message)
    {
        return split(dlib::tolower(message), " \t\n\r\v\f!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~");
    }

    static std::pair<sample_type, sample_type> count_ngrams(const std::vector<string>& tokens)
    {
        return {
            hash_word_ngrams(tokens, min_word_n, max_word_n, num_buckets),
            hash_char_ngrams(tokens, min_char_n, max_char_n, num_buckets)
        };
    }

    tfidf_transformer<sample_type> words = make_tfidf();
    tfidf_transformer<sample_type> chars = make_tfidf();
    // names[j] holds the n-grams of the training messages behind dimension j.
    std::vector<std::set<string>> names;
};

// ----------------------------------------------------------------------------------------

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("test-fraction", "fraction of messages kept for testing (default: 0.2)", 1);
    parser.add_option("c", "use this C instead of choosing it by cross-validation", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() < 1 || parser.option("h") || parser.option("help"))
    {
        cout << "This example needs the SMS Spam Collection to run." << endl;
        cout << "You can get it from https://archive.ics.uci.edu/dataset/228/sms+spam+collection" << endl;
        cout << "Uncompress it and give the path to the SMSSpamCollection file to this program." << endl;
        parser.print_options();
        return EXIT_SUCCESS;
    }
    parser.check_option_arg_range("test-fraction", 0.01, 0.99);
    parser.check_option_arg_range("c", 1e-6, 1e6);
    const double test_fraction = get_option(parser, "test-fraction", 0.2);

    std::vector<string> messages;
    std::vector<double> labels;
    load_sms_spam_collection(parser[0], messages, labels);
    const auto num_spam = std::count(labels.begin(), labels.end(), +1);
    cout << "# messages: " << messages.size() << " (" << num_spam << " spam)" << endl;

    // Shuffle before splitting, since nothing says the file's order is random.
    randomize_samples(messages, labels);
    const size_t num_test = lround(messages.size() * test_fraction);
    const std::vector<string> test_messages(messages.end() - num_test, messages.end());
    const std::vector<double> test_labels(labels.end() - num_test, labels.end());
    messages.resize(messages.size() - num_test);
    labels.resize(labels.size() - num_test);

    // Learn the TF-IDF weights from the training messages only.  Document frequencies are
    // information about the data set, and computing them on the test messages would make
    // the test accuracy optimistic.
    text_features features;
    features.train(messages);
    cout << features << endl;
    std::vector<sample_type> samples;
    for (const auto& message : messages)
        samples.push_back(features(message));

    // Now we train the classifier.  linear_classifier_trainer trains a linear SVM and:
    //   - gives each class a C inversely proportional to its number of messages, so the
    //     6 times more ham messages don't outweigh the spam ones,
    //   - picks C among the candidates by 5-fold cross-validation, printing each one's
    //     accuracy on spam (+1) and ham (-1),
    //   - turns the SVM's score into a probability of spam with Platt scaling.
    // The TF-IDF weights have seen every fold, a small leak that is fine for picking C,
    // which is why the final numbers come from the test set.
    linear_classifier_trainer<sample_type> trainer;
    trainer.set_num_folds(5);
    if (parser.option("c"))
        trainer.set_c(get_option(parser, "c", 1.0));
    else
        trainer.set_c_candidates({0.01, 0.1, 1.0, 10.0, 100.0});
    trainer.be_verbose();
    double c_used = 0;
    const auto df = trainer.train(samples, labels, c_used);
    cout << "using C = " << c_used << endl;

    // C = 0.01 underfits and misses more spam.  From 0.1 up, the candidates differ by a
    // few messages, within the noise of one shuffle: the classes are close to separable,
    // so past a point C barely matters.  A real filter
    // would pick C and the threshold knowing that losing a message costs more than
    // missing a spam.

    // A linear SVM is one weight per dimension, so we can see what it learned: the
    // dimensions with the largest weights push a message toward spam, and those with the
    // most negative weights toward ham.  describe() tells us which n-grams they stand for.
    std::vector<std::pair<double, unsigned long>> weights;
    for (const auto& [dimension, value] : df.decision_funct.basis_vectors(0))
        weights.emplace_back(value * df.decision_funct.alpha(0), dimension);
    std::sort(weights.begin(), weights.end());
    const auto print_dimensions = [&](const auto begin, const auto end)
    {
        for (auto it = begin; it != end; ++it)
        {
            cout << "  " << setw(6) << it->first << "  ";
            const char* separator = "";
            for (const auto& ngram : features.describe(it->second))
            {
                cout << separator << '"' << ngram << '"';
                separator = " | ";
            }
            cout << endl;
        }
    };
    cout << fixed << setprecision(3);
    cout << "\ndimensions pushing toward spam" << endl;
    print_dimensions(weights.rbegin(), weights.rbegin() + 10);
    cout << "\ndimensions pushing toward ham" << endl;
    print_dimensions(weights.begin(), weights.begin() + 10);

    // Now, let's see how the model does on the test messages, which neither the TF-IDF
    // weights nor the SVM have seen.  A message is spam if its probability is above 0.5.
    long true_spam = 0, false_spam = 0, true_ham = 0, false_ham = 0;
    for (size_t i = 0; i < test_messages.size(); ++i)
    {
        const bool predicted_spam = df(features(test_messages[i])) > 0.5;
        const bool is_spam = test_labels[i] > 0;
        if (predicted_spam && is_spam)
            ++true_spam;
        else if (predicted_spam)
            ++false_spam;
        else if (is_spam)
            ++false_ham;
        else
            ++true_ham;
    }
    // With the default settings, cross-validation picks C = 0.1, and the model gets
    // 98.7% of the 1115 test messages right: 5 ham messages are marked as spam and 10
    // spam messages are missed, i.e. a spam precision of 96.5% and a recall of 93.2%.
    cout << "\ntesting results on " << test_messages.size() << " messages" << endl;
    cout << "  accuracy:         " << double(true_spam + true_ham) / test_messages.size() << endl;
    cout << "  spam precision:   " << double(true_spam) / (true_spam + false_spam) << endl;
    cout << "  spam recall:      " << double(true_spam) / (true_spam + false_ham) << endl;
    cout << "  ham marked spam:  " << false_spam << endl;
    cout << "  spam missed:      " << false_ham << endl;

    // Finally, we save the model to disk.  Remember that the SVM's weights refer to the
    // dimensions of the TF-IDF vectors, so the features object must be saved with it.
    serialize("sms_spam_classifier.dat") << features << df;

    // And we can load it back and classify new messages.
    text_features features2;
    probabilistic_decision_function<kernel_type> df2;
    deserialize("sms_spam_classifier.dat") >> features2 >> df2;

    // The third message has "free" and "call me", but reads like a message between
    // friends, and gets 0.035.  The fourth has no prize or phone number, but "credited",
    // "bonus points" and "Reply STOP" sound automated, and it gets 0.997.
    cout << "\nprobability of spam for new messages" << endl;
    for (const string message : {
        "WINNER!! You have been selected to receive a £900 prize reward. Call 09061701461 to claim",
        "Are we still meeting for lunch tomorrow? I can be there at 1",
        "I have a free ticket for the game tonight, call me if you want it",
        "Your account has been credited with 500 bonus points. Reply STOP to opt out"})
    {
        cout << "  " << df2(features2(message)) << "  " << message << endl;
    }

    return EXIT_SUCCESS;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
