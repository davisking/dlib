// Copyright (C) 2025 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_BPE_TOKENIZER_ABSTRACT_
#ifdef DLIB_BPE_TOKENIZER_ABSTRACT_

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
#include <mutex>
#include <thread>
#include <future>
#include <queue>

namespace dlib
{

    class bpe_tokenizer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class implements a Byte Pair Encoding (BPE) tokenizer, which is a subword
                tokenization algorithm commonly used in natural language processing (NLP). The
                BPE algorithm iteratively merges the most frequent pairs of bytes or characters
                to form a vocabulary of subword units. This approach is particularly useful for
                handling out-of-vocabulary words and reducing the size of the vocabulary while
                maintaining the ability to represent any text.

                The tokenizer supports special tokens, which can be used to mark specific elements
                in the text (e.g., `<text>`, `<url>`, `<image>`, etc.). These special tokens are
                treated as atomic units during tokenization and are not subject to further splitting.

                The class provides methods for training the tokenizer on a given text corpus, encoding
                text into subword tokens, and decoding tokens back into text. The tokenizer can be
                serialized and deserialized to/from a file, allowing for easy storage and reuse.

                REFERENCES
                    - Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of
                      Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of
                      the Association for Computational Linguistics (ACL 2016).

            INITIAL VALUE
                - The base vocabulary is initialized with single-byte tokens (0-255).
                - Special tokens are pre-defined and assigned IDs starting from 256.
                - The maximum token length is set to 8 bytes.
        !*/

    public:
        bpe_tokenizer();
        /*!
            ensures
                - Initializes the tokenizer with a base vocabulary of single-byte tokens (0-255).
                - Pre-defines special tokens and assigns them unique IDs starting from 256.
        !*/

        void train(
            const std::string& text,
            int vocab_size,
            bool verbose = false
        );
        /*!
            requires
                - vocab_size >= 256
            ensures
                - Trains the tokenizer on the provided text corpus.
                - Iteratively merges the most frequent pairs of tokens to form a subword vocabulary
                  of size `vocab_size`.
                - If `verbose` is true, progress information is printed to the standard output.
        !*/

        std::vector<int> encode(
            const std::string& text
        ) const;
        /*!
            ensures
                - Encodes the input text into a sequence of subword tokens.
                - Special tokens are automatically added to mark the beginning and end of paragraphs.
                - Returns a vector of token IDs representing the encoded text.
        !*/

        std::string decode(
            const std::vector<int>& ids,
            bool display_special_tokens = true
        ) const;
        /*!
            ensures
                - Decodes a sequence of token IDs back into a human-readable string.
                - If `display_special_tokens` is true, special tokens are included in the output.
                - Returns the decoded text as a UTF-8 encoded string.
        !*/

        std::string decode(int id, bool display_special_tokens = true) const
        { return decode(std::vector<int>({ id }), display_special_tokens); }
        /*!
            ensures
                - decode a single token back into text.
        !*/

        int get_special_token_id(
            const std::string& token
        ) const;
        /*!
            ensures
                - Returns the ID of the specified special token.
                - Throws an exception if the token is not found in the special tokens map.
        !*/

        size_t get_vocab_size() const;
        /*!
            ensures
                - Returns the total size of the vocabulary, including base tokens and special tokens.
        !*/
    };

    void serialize(
        const bpe_tokenizer& tok,
        std::ostream& out
    );
    /*!
        ensures
            - Saves the entire state of tok to out.
    !*/

    void deserialize(
        bpe_tokenizer& tok,
        std::istream& in
    );
    /*!
        ensures
            - Restores the state of a bpe_tokenizer from a serialized state.
    !*/
}

#endif // DLIB_BPE_TOKENIZER_ABSTRACT_
