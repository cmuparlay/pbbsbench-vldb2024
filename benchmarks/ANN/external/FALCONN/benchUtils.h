#ifndef __BENCHUTILS_H__
#define __BENCHUTILS_H__

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "parse_command_line.h"

enum class ptr_mapped_src{
  NATIVE, VOLATILE, PERSISTENT, TRANSITIVE
};

namespace detail{

template<typename T, ptr_mapped_src Src>
class ptr_mapped_impl
{
  T *ptr_raw;
public:
  using difference_type = std::ptrdiff_t;
  using value_type = std::remove_cv_t<T>;
  using pointer = T*;
  using reference = T&;
  using iterator_category = std::random_access_iterator_tag;

  ptr_mapped_impl(){
  }

  ptr_mapped_impl(T *p) : ptr_raw(p){
  }

  template<typename U, ptr_mapped_src SrcOther>
  ptr_mapped_impl(const ptr_mapped_impl<U,SrcOther> &ptr) :
    ptr_raw(ptr.get()){
    static_assert(std::is_convertible_v<U*,T*>);
  }

  ptr_mapped_impl& operator=(T *p){
    ptr_raw = p;
    return *this;
  }

  template<typename U, ptr_mapped_src SrcOther>
  ptr_mapped_impl& operator=(const ptr_mapped_impl<U,SrcOther> &ptr){
    static_assert(std::is_convertible_v<U*,T*>);
    ptr_raw = ptr.get();
  }

  T* get() const{
    return ptr_raw;
  }

  operator T*() const{
    return get();
  }

  // For simplicity, we only keep the least methods to satisfy the requirements of LegacyIterator

  T& operator*() const{
    return *get();
  }

  ptr_mapped_impl& operator++(){
    ++ptr_raw;
    return *this;
  }

  ptr_mapped_impl& operator+=(size_t n){
    ptr_raw += n;
    return *this;
  }

  ptr_mapped_impl operator+(size_t n) const{
    return ptr_raw+n;
  }

  ptr_mapped_impl& operator-=(size_t n){
    ptr_raw -= n;
    return *this;
  }

  ptr_mapped_impl operator-(size_t n) const{
    return ptr_raw - n;
  }

  difference_type operator-(const ptr_mapped_impl &other) const{
    return ptr_raw - other.ptr_raw;
  }

  reference operator[](size_t i) const{
    return ptr_raw[i];
  }

  bool operator<(const ptr_mapped_impl &other) const{
    return ptr_raw < other.ptr_raw;
  }

  bool operator>(const ptr_mapped_impl &other) const{
    return other<*this;
  }

  bool operator>=(const ptr_mapped_impl &other) const{
    return !(*this<other);
  }

  bool operator<=(const ptr_mapped_impl &other) const{
    return !(*this>other);
  }
};

} // namespace detail

template<typename T, ptr_mapped_src Src>
using ptr_mapped = std::conditional_t<Src==ptr_mapped_src::NATIVE, T*, detail::ptr_mapped_impl<T,Src>>;
/*
template<typename T, ptr_mapped_src Src>
struct std::iterator_traits<detail::ptr_mapped_impl<T,Src>>
{
  using difference_type = std::ptrdiff_t;
  using value_type = std::remove_cv_t<T>;
  using pointer = T*;
  using reference = T&;
  using iterator_category = void;
};
*/
// *************************************************************
//  SOME DEFINITIONS
// *************************************************************


// *************************************************************
// Parsing code (should move to common?)
// *************************************************************

// returns a pointer and a length
std::pair<char*, size_t> mmapStringFromFile(const char* filename) {
  struct stat sb;
  int fd = open(filename, O_RDONLY);
  if (fd == -1) {
    perror("open");
    exit(-1);
  }
  if (fstat(fd, &sb) == -1) {
    perror("fstat");
    exit(-1);
  }
  if (!S_ISREG(sb.st_mode)) {
    perror("not a file\n");
    exit(-1);
  }
  char* p =
      static_cast<char*>(mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (p == MAP_FAILED) {
    perror("mmap");
    exit(-1);
  }
  if (close(fd) == -1) {
    perror("close");
    exit(-1);
  }
  size_t n = sb.st_size;
  return std::make_pair(p, n);
}

#endif // __BENCHUTILS_H__