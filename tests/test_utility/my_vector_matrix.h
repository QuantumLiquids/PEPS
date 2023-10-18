//
// Created by haoxinwang on 17/10/2023.
//

#ifndef GRACEQ_VMC_PEPS_MY_VECTOR_MATRIX_H
#define GRACEQ_VMC_PEPS_MY_VECTOR_MATRIX_H

#include "gqten/gqten.h"

using namespace gqten;

using std::vector;

GQTEN_Complex MyConj(GQTEN_Complex a) {
  return conj(a);
}

GQTEN_Double MyConj(GQTEN_Double a) {
  return a;
}

template<typename T>
class MyVector {
 public:
  MyVector() = default;

  MyVector(size_t length) : elements_(length) {}

  explicit MyVector(const std::vector<T> &elements) : elements_(elements) {}

  explicit MyVector(const std::vector<T> &&elements) : elements_(std::move(elements)) {}

  // these two get/set functions used in MPI transforming data
  std::vector<T> &GetElements() {
    return elements_;
  }

  const std::vector<T> &GetElements() const {
    return elements_;
  }

  int GetSize() const {
    return elements_.size();
  }


  MyVector<T> operator+(const MyVector<T> &rhs) const {
    MyVector<T> result(GetSize());
    const std::vector<T> &otherElements = rhs.GetElements();

    if (GetSize() != rhs.GetSize()) {
      throw std::runtime_error("Vector dimensions do not match.");
    }

    for (int i = 0; i < GetSize(); i++) {
      result[i] = elements_[i] + otherElements[i];
    }

    return result;
  }

  MyVector<T> &operator+=(const MyVector<T> &rhs) {
    const std::vector<T> &otherElements = rhs.GetElements();

    if (GetSize() != rhs.GetSize()) {
      throw std::runtime_error("Vector dimensions do not match.");
    }

    for (int i = 0; i < GetSize(); i++) {
      elements_[i] += otherElements[i];
    }
    return *this;
  }

  MyVector<T> operator-(const MyVector<T> &rhs) const {
    MyVector<T> result(GetSize());
    std::vector<T> otherElements = rhs.GetElements();
    if (GetSize() != rhs.GetSize()) {
      throw std::runtime_error("Vector dimensions do not match.");
    }
    for (int i = 0; i < GetSize(); i++) {
      result[i] = elements_[i] - otherElements[i];
    }
    return result;
  }

  template<typename U>
  MyVector<T> operator*(U scalar) const {
    MyVector<T> result(elements_.size());
    for (size_t i = 0; i < elements_.size(); i++) {
      result[i] = elements_[i] * scalar;
    }
    return result;
  }

  T operator*(const MyVector<T> &other) const {
    std::vector<T> otherElements = other.GetElements();

    if (GetSize() != other.GetSize()) {
      throw std::runtime_error("Vector dimensions do not match.");
    }

    T result = T(0);

    for (int i = 0; i < GetSize(); i++) {
      result += MyConj(elements_[i]) * otherElements[i];
    }

    return result;
  }

  double Norm() const {
    double res = 0;
    for (auto elem: elements_) {
      res += std::abs(elem) * std::abs(elem);
    }
    return res;
  }

  void Print() const {
    std::cout << "[";
    for (const auto &element: elements_) {
      std::cout << " " << element;
    }
    std::cout << "]" << std::endl;
  }

 private:
  std::vector<T> elements_;

  T &operator[](size_t index) {
    if (index >= GetSize()) {
      throw std::out_of_range("Index out of range.");
    }

    return elements_[index];
  }

  const T &operator[](size_t index) const {
    if (index >= GetSize()) {
      throw std::out_of_range("Index out of range.");
    }

    return elements_[index];
  }

};

template<typename T>
MyVector<T> operator*(const T scalar, const MyVector<T> &other) {
  return other * scalar;
}

MyVector<GQTEN_Complex> operator*(const double scalar, const MyVector<GQTEN_Complex> &other) {
  return other * GQTEN_Complex(scalar, 0.0);
}


template<typename ElemT>
class MySquareMatrix {
 public:
  MySquareMatrix() = default;

  MySquareMatrix(size_t length) :
      data_(length, vector<ElemT>(length, ElemT(0))) {}

  MySquareMatrix(const MySquareMatrix &rhs) : data_(rhs.data_) {}

  explicit MySquareMatrix(std::vector<std::vector<ElemT>> matrix) : data_(matrix) {}

  const std::vector<std::vector<ElemT>> &GetMatrix() const {
    return data_;
  }

  MyVector<ElemT> operator*(const MyVector<ElemT> &vector) const {
    std::vector<ElemT> vectorElements = vector.GetElements();


    if (Length() != vector.GetSize()) {
      assert(Length() == vector.GetSize());
      throw std::runtime_error("Matrix and vector dimensions do not match.");
    }

    std::vector<ElemT> resultElements(Length(), ElemT(0));

    for (int i = 0; i < Length(); i++) {
      for (int j = 0; j < Length(); j++) {
        resultElements[i] += data_[i][j] * vectorElements[j];
      }
    }
    return MyVector<ElemT>(resultElements);
  }

  size_t Length() const { return data_.size(); };


 private:
  std::vector<std::vector<ElemT>> data_;
};


#endif //GRACEQ_VMC_PEPS_MY_VECTOR_MATRIX_H
