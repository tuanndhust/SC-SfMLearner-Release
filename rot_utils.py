import numpy as np
import math

_FLOAT_EPS_4 = np.finfo(float).eps * 4.0

def mat2euler(M, cy_thresh=None, seq='zyx'):
	'''
	Taken Forom: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
	Discover Euler angle vector from 3x3 matrix
	Uses the conventions above.
	Parameters
	----------
	M : array-like, shape (3,3)
	cy_thresh : None or scalar, optional
		 threshold below which to give up on straightforward arctan for
		 estimating x rotation.  If None (default), estimate from
		 precision of input.
	Returns
	-------
	z : scalar
	y : scalar
	x : scalar
		 Rotations in radians around z, y, x axes, respectively
	Notes
	-----
	If there was no numerical error, the routine could be derived using
	Sympy expression for z then y then x rotation matrix, which is::
		[                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
		[cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
		[sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
	with the obvious derivations for z, y, and x
		 z = atan2(-r12, r11)
		 y = asin(r13)
		 x = atan2(-r23, r33)
	for x,y,z order
		y = asin(-r31)
		x = atan2(r32, r33)
    z = atan2(r21, r11)
	Problems arise when cos(y) is close to zero, because both of::
		 z = atan2(cos(y)*sin(z), cos(y)*cos(z))
		 x = atan2(cos(y)*sin(x), cos(x)*cos(y))
	will be close to atan2(0, 0), and highly unstable.
	The ``cy`` fix for numerical instability below is from: *Graphics
	Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
	0123361559.  Specifically it comes from EulerAngles.c by Ken
	Shoemake, and deals with the case where cos(y) is close to zero:
	See: http://www.graphicsgems.org/
	The code appears to be licensed (from the website) as "can be used
	without restrictions".
	'''
	M = np.asarray(M)
	if cy_thresh is None:
			try:
					cy_thresh = np.finfo(M.dtype).eps * 4
			except ValueError:
					cy_thresh = _FLOAT_EPS_4
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
	# cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
	cy = math.sqrt(r33*r33 + r23*r23)
	if seq=='zyx':
		if cy > cy_thresh: # cos(y) not close to zero, standard form
				z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
				y = math.atan2(r13,  cy) # atan2(sin(y), cy)
				x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
		else: # cos(y) (close to) zero, so x -> 0.0 (see above)
				# so r21 -> sin(z), r22 -> cos(z) and
				z = math.atan2(r21,  r22)
				y = math.atan2(r13,  cy) # atan2(sin(y), cy)
				x = 0.0
	elif seq=='xyz':
		if cy > cy_thresh:
			y = math.atan2(-r31, cy)
			x = math.atan2(r32, r33)
			z = math.atan2(r21, r11)
		else:
			z = 0.0
			if r31 < 0:
				y = np.pi/2
				x = math.atan2(r12, r13)
			else:
				y = -np.pi/2
				#x =
	else:
		raise Exception('Sequence not recognized')
	return z, y, x


def euler2quat(z=0, y=0, x=0, isRadian=True):
	''' Return quaternion corresponding to these Euler angles
	Uses the z, then y, then x convention above
	Parameters
	----------
	z : scalar
		 Rotation angle in radians around z-axis (performed first)
	y : scalar
		 Rotation angle in radians around y-axis
	x : scalar
		 Rotation angle in radians around x-axis (performed last)
	Returns
	-------
	quat : array shape (4,)
		 Quaternion in w, x, y z (real, then vector) format
	Notes
	-----
	We can derive this formula in Sympy using:
	1. Formula giving quaternion corresponding to rotation of theta radians
		 about arbitrary axis:
		 http://mathworld.wolfram.com/EulerParameters.html
	2. Generated formulae from 1.) for quaternions corresponding to
		 theta radians rotations about ``x, y, z`` axes
	3. Apply quaternion multiplication formula -
		 http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
		 formulae from 2.) to give formula for combined rotations.
	'''

	if not isRadian:
		z = ((np.pi) / 180.) * z
		y = ((np.pi) / 180.) * y
		x = ((np.pi) / 180.) * x
	z = z / 2.0
	y = y / 2.0
	x = x / 2.0
	cz = math.cos(z)
	sz = math.sin(z)
	cy = math.cos(y)
	sy = math.sin(y)
	cx = math.cos(x)
	sx = math.sin(x)
	return np.array([
		cx * cy * cz - sx * sy * sz,
		cx * sy * sz + cy * cz * sx,
		cx * cz * sy - sx * cy * sz,
		cx * cy * sz + sx * cz * sy])



