
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
#include <sstream>
#include <string>

namespace fs = std::filesystem;

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>

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
void save_to_file_xml ( std::string && object_name_, T const & t_, fs::path && path_, std::string && file_name_,
                        bool const append_ = false ) noexcept {
    std::ofstream ostream ( path_ / ( file_name_ + std::string ( ".xmlcereal" ) ),
                            append_ ? std::ios::app | std::ios::out : std::ios::out );
    {
        cereal::XMLOutputArchive archive ( ostream );
        archive ( cereal::make_nvp ( object_name_, t_ ) );
    }
    ostream.flush ( );
    ostream.close ( );
}

template<typename T>
void load_from_file_xml ( std::string && object_name_, T & t_, fs::path && path_, std::string && file_name_ ) noexcept {
    std::ifstream istream ( path_ / ( file_name_ + std::string ( ".xmlcereal" ) ) );
    {
        cereal::XMLInputArchive archive ( istream );
        archive ( cereal::make_nvp ( object_name_, t_ ) );
    }
    istream.close ( );
}

template<typename T>
void save_to_file_json ( std::string && object_name_, T const & t_, fs::path && path_, std::string && file_name_, bool const append_ = false ) noexcept {
    std::ofstream ostream ( path_ / ( file_name_ + std::string ( ".json" ) ),
                            append_ ? std::ios::app | std::ios::out : std::ios::out );
    {
        cereal::JSONOutputArchive archive ( ostream );
        archive ( cereal::make_nvp ( object_name_, t_ ) );
    }
    ostream.flush ( );
    ostream.close ( );
}

template<typename T>
void load_from_file_json ( std::string && object_name_, T & t_, fs::path && path_, std::string && file_name_ ) noexcept {
    std::ifstream istream ( path_ / ( file_name_ + std::string ( ".json" ) ) );
    {
        cereal::JSONInputArchive archive ( istream );
        archive ( cereal::make_nvp ( object_name_, t_ ) );
    }
    istream.close ( );
}

[[nodiscard]] std::string get_timestamp_utc ( ) noexcept;
[[nodiscard]] std::string get_timestamp ( ) noexcept;

void sleep_for_milliseconds ( std::int32_t const milliseconds_ ) noexcept;

void cls ( ) noexcept;
// x is the column, y is the row. The origin (0,0) is top-left.
void set_cursor_position ( int x_, int y_ ) noexcept;

void write_buffer ( std::wostringstream const & outbuf_ ) noexcept;

[[nodiscard]] bool hide_cursor ( ) noexcept;
