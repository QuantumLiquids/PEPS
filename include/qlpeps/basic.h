// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-03
*
* Description: QuantumLiquids/PEPS project. Basic structures and classes.
*/

#ifndef QLPEPS_BASIC_H
#define QLPEPS_BASIC_H

#include <string>     // string
#include <vector>     // vector
#include <random>     // default_random_engine

namespace qlpeps {


enum BondOrientation {
  HORIZONTAL = 0,
  VERTICAL
};

BondOrientation Rotate(BondOrientation orient) {
  return BondOrientation(1 - (size_t) orient);
}

///<
/**from which direction, or the position
 *
 * UP:   MPS tensors are numbered from right to left
               2--t--0
                 |
                 1

 * DOWN: MPS tensors are numbered from left to right
                   1
                   |
                0--t--2

 * LEFT: MPS tensors are numbered from up to down;
 *
 * the order, left, down, right, up, follow the MPO/single layer tps indexes order
 */
enum BMPSPOSITION {
  LEFT = 0,
  DOWN,
  RIGHT,
  UP
};

/**
 * LEFT/RIGHT -> HORIZONTAL
 * UP/DOWN -> VERTICAL
 * @param post
 * @return
 */
BondOrientation Orientation(const BMPSPOSITION post) {
  return static_cast<enum BondOrientation>((static_cast<size_t>(post) % 2));
}

size_t MPOIndex(const BMPSPOSITION post) {
  return static_cast<size_t>(post);
}

BMPSPOSITION Opposite(const BMPSPOSITION post) {
  return static_cast<BMPSPOSITION>((static_cast<size_t>(post) + 2) % 4);
  switch (post) {
    case DOWN:return UP;
    case UP:return DOWN;
    case LEFT:return RIGHT;
    case RIGHT:return LEFT;
  }
}

enum DIAGONAL_DIR {
  LEFTUP_TO_RIGHTDOWN,
  LEFTDOWN_TO_RIGHTUP
};

}

#endif //QLPEPS_BASIC_H
