
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

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <fcntl.h>
#include <io.h>

#include <cstdio>
#include <cstdlib>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

[[nodiscard]] fs::path appDataPath ( std::string && name_ ) {
    char * value;
    std::size_t len;
    _dupenv_s ( &value, &len, "USERPROFILE" );
    fs::path return_value ( std::string ( value ) + std::string ( "\\AppData\\Roaming\\" + name_ ) );
    fs::create_directory ( return_value ); // No error if directory exists.
    return return_value;
}

fs::path app_data_path_          = appDataPath ( "SimdNet" );
fs::path const & g_app_data_path = app_data_path_;

[[nodiscard]] fs::path getExePath ( ) noexcept {
    TCHAR exename[ 1024 ];
    GetModuleFileName ( NULL, exename, 1'024 );
    return fs::path ( exename ).parent_path ( );
}

fs::path app_path_          = getExePath ( );
fs::path const & g_app_path = app_path_;

std::string get_timestamp_utc ( ) noexcept {
    time_t rawtime = time ( NULL );
    struct tm ptm;
    gmtime_s ( &ptm, &rawtime );
    char buffer[ 32 ]{};
    std::snprintf ( buffer, 32, "%4i%02i%02i%02i%02i%02i", ptm.tm_year + 1900, ptm.tm_mon + 1, ptm.tm_mday, ptm.tm_hour, ptm.tm_min,
                    ptm.tm_sec );
    return { buffer };
}

std::string get_timestamp ( ) noexcept {
    time_t rawtime = time ( NULL );
    struct tm ptm;
    localtime_s ( &ptm, &rawtime );
    char buffer[ 32 ]{};
    std::snprintf ( buffer, 32, "%4i%02i%02i%02i%02i%02i", ptm.tm_year + 1900, ptm.tm_mon + 1, ptm.tm_mday, ptm.tm_hour, ptm.tm_min,
                    ptm.tm_sec );
    return { buffer };
}

void sleep_for_milliseconds ( std::int32_t const milliseconds_ ) noexcept {
    std::this_thread::sleep_for ( std::chrono::milliseconds ( milliseconds_ ) );
}

// https : // stackoverflow.com/questions/34842526/update-console-without-flickering-c

void cls ( ) noexcept {
    // Get the Win32 handle representing standard output.
    // This generally only has to be done once, so we make it static.
    static HANDLE const hOut = GetStdHandle ( STD_OUTPUT_HANDLE );
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    COORD topLeft = { 0, 0 };
    // std::cout uses a buffer to batch writes to the underlying console.
    // We need to flush that to the console because we're circumventing
    // std::cout entirely; after we clear the console, we don't want
    // stale buffered text to randomly be written out.
    std::wcout.flush ( );
    // Figure out the current width and height of the console window.
    if ( not GetConsoleScreenBufferInfo ( hOut, &csbi ) )
        std::abort ( );
    DWORD length = csbi.dwSize.X * csbi.dwSize.Y;
    DWORD written;
    // Flood-fill the console with spaces to clear it.
    FillConsoleOutputCharacter ( hOut, TEXT ( ' ' ), length, topLeft, &written );
    // Reset the attributes of every character to the default.
    // This clears all background colour formatting, if any.
    FillConsoleOutputAttribute ( hOut, csbi.wAttributes, length, topLeft, &written );
    // Move the cursor back to the top left for the next sequence of writes.
    SetConsoleCursorPosition ( hOut, topLeft );
}

// x is the column, y is the row. The origin (0, 0) is top-left.
void set_cursor_position ( int x_, int y_ ) noexcept {
    static HANDLE const hOut = GetStdHandle ( STD_OUTPUT_HANDLE );
    std::wcout.flush ( );
    COORD coord = { ( SHORT ) x_, ( SHORT ) y_ };
    SetConsoleCursorPosition ( hOut, coord );
}

void write_buffer ( std::wostringstream const & outbuf_ ) noexcept {
    static HANDLE const hOut = GetStdHandle ( STD_OUTPUT_HANDLE );
    static COORD topLeft     = { 0, 0 };
    DWORD dwCharsWritten; // <-- this might not be necessary
    // You might be able to pass in NULL if you don't want to keep track of the
    // number of characters written. Some functions allow you to do this, others
    // don't. I'm not 100% sure about this one, the documentation doesn't say.
    WriteConsoleOutputCharacter ( hOut, outbuf_.str ( ).c_str ( ), outbuf_.str ( ).length ( ), topLeft, &dwCharsWritten );
}

void set_mode_unicode ( ) noexcept { _setmode ( _fileno ( stdout ), _O_U16TEXT ); }

bool hide_cursor ( ) noexcept {
    static HANDLE const hOut = GetStdHandle ( STD_OUTPUT_HANDLE );
    static CONSOLE_CURSOR_INFO const info{ 1, false };
    set_mode_unicode ( );
    return SetConsoleCursorInfo ( hOut, &info );
}
