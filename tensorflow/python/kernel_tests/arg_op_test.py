# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for tensorflow.ops.math_ops.arg"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
from cwise_ops_test import UnaryOpTest


class ArgOpTest(tf.test.TestCase):

  def _compareArg(self, x):
    with self.test_session() as sess:
      res = sess.run(tf.arg(x))
      compare = map(float, np.angle(x))
      self.assertAllClose(res, compare)

  def testSingleRealNum(self):
    x = -0.5
    y = 0.0
    z = 0.5

    self._compareArg([x])
    self._compareArg([y])
    self._compareArg([z])

  def testListRealNums(self):
    x = [10.0* random.random() for a in xrange(-100,100)]
    self._compareArg(x)

  def testListComplexNums(self):
    r = np.array([10 * random.random() for a in xrange(-100,100)])
    i = np.array([10 * random.random() for a in xrange(-100,100)])
    c = r + i * 1j
    d = map(np.complex64, c)

    self._compareArg(d)

  def testCrazyTheory(self):
    tester = UnaryOpTest()
    r = np.array([10 * random.random() for a in xrange(-100,100)])
    i = np.array([10 * random.random() for a in xrange(-100,100)])
    c = r + i * 1j
    d = map(np.complex64, c)
    tester._compareCpu(d, np.angle, tf.arg)




if __name__ == "__main__":
  tf.test.main()
