
// MIT License
//
// Copyright (c) 2019 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>

extern fs::path const & g_app_data_path;
extern fs::path const & g_app_path;

template<typename T>
void save_to_file_bin ( T const & t_, fs::path && path_, std::string && file_name_, bool const append_ = false ) noexcept {
    std::ofstream ostream ( path_ / ( file_name_ + std::string ( ".cereal" ) ),
                            append_ ? std::ios::binary | std::ios::app | std::ios::out : std::ios::binary | std::ios::out );
    {
        cereal::BinaryOutputArchive archive ( ostream );
        archive ( t_ );
    }
    ostream.flush ( );
    ostream.close ( );
}

template<typename T>
void load_from_file_bin ( T & t_, fs::path && path_, std::string && file_name_ ) noexcept {
    std::ifstream istream ( path_ / ( file_name_ + std::string ( ".cereal" ) ), std::ios::binary );
    {
        cereal::BinaryInputArchive archive ( istream );
        archive ( t_ );
    }
    istream.close ( );
}

template<typename T>
void save_to_file_xml ( T const & t_, fs::path && path_, std::string && file_name_, bool const append_ = false ) noexcept {
    std::ofstream ostream ( path_ / ( file_name_ + std::string ( ".xmlcereal" ) ),
                            append_ ? std::ios::app | std::ios::out : std::ios::out );
    {
        cereal::XMLOutputArchive archive ( ostream );
        archive ( t_ );
    }
    ostream.flush ( );
    ostream.close ( );
}

template<typename T>
void load_from_file_xml ( T & t_, fs::path && path_, std::string && file_name_ ) noexcept {
    std::ifstream istream ( path_ / ( file_name_ + std::string ( ".xmlcereal" ) ) );
    {
        cereal::XMLInputArchive archive ( istream );
        archive ( t_ );
    }
    istream.close ( );
}
