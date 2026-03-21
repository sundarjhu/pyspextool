;+
; NAME:
;     mc_readwavecalinfo
;
; PURPOSE:
;     To read a Spextool wavecalinfo file.
;
; CALLING SEQUENCE:
;     result = mc_readwavecalinfo(file,CANCEL=cancel)
;
; INPUTS:
;     file - A Spextool wavecal info file.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return if there is a problem.
;
; OUTPUTS:
;     A structure with the tags for each parameter.
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;     'file' must be a Spextool wavecalinfo file.
;
; DEPENDENCIES:
;     Spextool library (and its dependecies)
;
; PROCEDURE:
;     Just parses a text file and creates a structure.
;
; EXAMPLES:
;     Duh.
;
; MODIFICATION HISTORY:
;     2017-10-08 - Written by M. Cushing, University of Toledo
;     2018-03-01 - Added the 1D case
;-
function mc_readwavecalinfo,file,CANCEL=cancel
  
  cancel = 0

;  Check parameters

  if n_params() lt 1 then begin
     
     print, 'Syntax - mc_readwavecalinfo,file,CANCEL=cancel'
     cancel = 1
     return,-1
     
  endif
  
  cancel = mc_cpar('mc_readwavecalinfo',file,1,'File',7,0)
  if cancel then return,-1

;  Get going
  
  wspec = readfits(file,hdr,/SILENT)

  naps = fxpar(hdr,'NAPS')
  norders = fxpar(hdr,'NORDERS')
  orders  = fix(mc_fsextract(fxpar(hdr,'ORDERS'),/INDEX))

  wcaltype = strtrim(fxpar(hdr,'WCALTYPE'),2)
  linelist = strtrim(fxpar(hdr,'LINELIST'),2)

;  Get extraction ranges

  xranges = intarr(2,norders)
  for i = 0,norders-1 do begin

     name = 'OR'+string(orders[i],FORMAT='(I3.3)')+'_XR'
     xranges[*,i] = long( strsplit( fxpar(hdr,name), ',', /EXTRACT) )

  endfor

  extap = fxpar(hdr,'EXTAP')

;  Get wavelength range of mode

  wmin = min(wspec[*,0,*],/NAN,MAX=wmax)
  
;  Get the output structure started
  
  s = {wspec:wspec,naps:naps,norders:norders,orders:orders, $
       xranges:xranges,wrange:[wmin,wmax],wcaltype:wcaltype, $
       linelist:linelist,extap:extap}

  if wcaltype eq '2D' then begin

     linedeg = fxpar(hdr,'LINEDEG')
     fndystep = fxpar(hdr,'FNDYSTEP')
     fndysum = fxpar(hdr,'FNDYSUM')
     genystep = fxpar(hdr,'GENYSTEP')
     c1xdeg = fxpar(hdr,'C1XDEG')

     s = create_struct(s, 'linedeg',linedeg[0],'fndystep',fndystep[0], $
                       'fndysum',fndysum[0],'genystep',genystep[0],$
                       'c1xdeg',c1xdeg[0])
     
     if linedeg eq 2 then begin
        
        c2xdeg = fxpar(hdr,'C2XDEG')
        
        s = create_struct(s, 'c2xdeg',c2xdeg[0])
        
     endif

     dispdeg   = fxpar(hdr,'DISPDEG')
     xcororder = fxpar(hdr,'XCORORDR')

;  Get the spectrum
     
     z = where(orders eq xcororder)

     xcorspec = reform(wspec[*,*,z])

     npixels = xranges[1,z]-xranges[0,z]+1
     xcorspec[*,0] = findgen(npixels)+total(xranges[0,z])

     ncoeffs = (dispdeg+1)
     p2wcoeffs = dblarr(ncoeffs)

     for i = 0, ncoeffs-1 do begin
        
        key = 'P2W_C'+string(i,format='(i2.2)')
        p2wcoeffs[i] = fxpar(hdr,key)
        
     endfor

     rms = fxpar(hdr,'FITRMS')
     
     s = create_struct(s,'dispdeg',dispdeg[0],'xcororder',xcororder[0], $
                       'xcorspec',xcorspec,'p2wcoeffs',p2wcoeffs,'rms',rms)
    
  endif
  
  if wcaltype eq '1D' then begin

     dispdeg   = fxpar(hdr,'DISPDEG')
     xcororder = fxpar(hdr,'XCORORDR')

;  Get the spectrum
     
     z = where(orders eq xcororder)

     xcorspec = reform(wspec[*,*,z])

     npixels = xranges[1,z]-xranges[0,z]+1
     xcorspec[*,0] = findgen(npixels)+total(xranges[0,z])

     ncoeffs = (dispdeg+1)
     p2wcoeffs = dblarr(ncoeffs)

     for i = 0, ncoeffs-1 do begin
        
        key = 'P2W_C'+string(i,format='(i2.2)')
        p2wcoeffs[i] = fxpar(hdr,key)
        
     endfor

     rms = fxpar(hdr,'FITRMS')
     
     s = create_struct(s,'dispdeg',dispdeg[0],'xcororder',xcororder[0], $
                       'xcorspec',xcorspec,'p2wcoeffs',p2wcoeffs,'rms',rms)


  endif
  
  if wcaltype eq '1DXD' or wcaltype eq '2DXD' then begin

     homeorder = fxpar(hdr,'HOMEORDR')     
     dispdeg   = fxpar(hdr,'DISPDEG')
     ordrdeg   = fxpar(hdr,'ORDRDEG')
     xcororder = fxpar(hdr,'XCORORDR')

;  Get the spectrum
     
     z = where(orders eq xcororder)
     xcorspec = reform(wspec[*,*,z])

     npixels = xranges[1,z]-xranges[0,z]+1
     xcorspec[*,0] = findgen(npixels)+total(xranges[0,z])

     
     ncoeffs = (dispdeg+1)*(ordrdeg+1)     
     p2wcoeffs = dblarr(ncoeffs)

     for i = 0, ncoeffs-1 do begin
        
        key = 'P2W_C'+string(i,format='(i2.2)')
        p2wcoeffs[i] = fxpar(hdr,key)
        
     endfor

     s = create_struct(s,'homeorder',homeorder[0],'dispdeg',dispdeg[0], $
                       'ordrdeg',ordrdeg[0],'xcororder',xcororder[0], $
                       'xcorspec',xcorspec,'p2wcoeffs',p2wcoeffs)
     
  endif

  if wcaltype eq '2DXD' then begin

     linedeg = fxpar(hdr,'LINEDEG')
     fndystep = fxpar(hdr,'FNDYSTEP')
     fndysum = fxpar(hdr,'FNDYSUM')
     genystep = fxpar(hdr,'GENYSTEP')
     c1xdeg = fxpar(hdr,'C1XDEG')
     c1ydeg = fxpar(hdr,'C1YDEG')

     s = create_struct(s, 'linedeg',linedeg[0],'fndystep',fndystep[0], $
                       'fndysum',fndysum[0],'genystep',genystep[0],$
                       'c1xdeg',c1xdeg[0],'c1ydeg',c1ydeg[0])
     
     if linedeg eq 2 then begin
        
        c2xdeg = fxpar(hdr,'C2XDEG')
        c2ydeg = fxpar(hdr,'C2YDEG')
        
        s = create_struct(s, 'c2xdeg',c2xdeg[0],'c2ydeg',c2ydeg[0])
        
     endif

  endif

  wavefmt = fxpar(hdr,'WAVEFMT')
  spatfmt = fxpar(hdr,'SPATFMT')
  
  s = create_struct(s, 'wavefmt',wavefmt,'spatfmt',spatfmt)
   
  return, s


end
