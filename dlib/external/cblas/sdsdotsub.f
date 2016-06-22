c     sdsdotsub.f
c
c     The program is a fortran wrapper for sdsdot.
c     Witten by Keita Teranishi.  2/11/1998
c
      subroutine sdsdotsub(n,x,incx,y,incy,dot)
c
      external sdsdot
      real sdsdot,dot
      integer n,incx,incy
      real x(*),y(*)
c
      dot=sdsdot(n,x,incx,y,incy)
      return
      end      
