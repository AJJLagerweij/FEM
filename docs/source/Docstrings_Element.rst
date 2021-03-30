*******************
Elements and Meshes
*******************
.. automodule:: element

Base Mesh
---------
.. autoclass:: element.Mesh
	:members:

    
Implemented Meshes
------------------
.. autoclass:: element.Mesh1D(x_start, x_end, num_ele, order, num_q, periodic=False)	

	.. automethod:: get_element
	.. automethod:: x_to_xi
	.. automethod:: xi_to_x
	.. automethod:: jacobian
	.. automethod:: shape
	.. automethod:: dshape

