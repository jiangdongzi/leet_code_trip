ass Solution {
	public double findMedianSortedArrays(int[] nums1, int[] nums2) {
		if (nums1 == null || nums1.length == 0) {
			if ((nums2.length & 1) == 0) {
				return nums2[nums2.length / 2] / 2.0 + nums2[nums2.length / 2 - 1] / 2.0;
			} else {
				return nums2[nums2.length / 2];
			}
		}

		if (nums2 == null || nums2.length == 0) {
			if ((nums1.length & 1) == 0) {
				return nums1[nums1.length / 2] / 2.0 + nums1[nums1.length / 2 - 1] / 2.0;
			} else {
				return nums1[nums1.length / 2];
			}
		}

		if (nums1[nums1.length - 1] <= nums2[0]) {
			if (nums1.length < nums2.length) {
				if ((nums1.length + nums2.length) % 2 == 0) {
					return nums2[(nums2.length - nums1.length) / 2] / 2.0 + nums2[(nums2.length - nums1.length) / 2 - 1] / 2.0;
				} else {
					return nums2[(nums2.length - nums1.length) / 2];
				}
			} else if (nums1.length == nums2.length) {
				return nums1[nums1.length - 1] / 2.0 + nums2[0] / 2.0;
			} else {
				if ((nums1.length + nums2.length) % 2 == 0) {
					return nums1[(nums1.length + nums2.length) / 2] / 2.0 + nums1[(nums1.length + nums2.length) / 2 - 1] / 2.0;
				} else {
					return nums1[(nums1.length + nums2.length) / 2];
				}
			}
		}

		return 1;
	}
}
