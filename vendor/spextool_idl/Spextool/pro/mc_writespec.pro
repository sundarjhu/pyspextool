;+
; NAME:
;     mc_writespec
;    
; PURPOSE:
;     Writes a SpeX spectra to disk
;   
; CATEGORY:
;     Spectroscopy
; 
; CALLING SEQUENCE:
;     mc_writespec,spectra,xranges,fullpath,aimage,sky,flat,naps,orders, $
;                  hdrinfo,appos,apradius,ps,slith_pix,slith_arc,slitw_pix, $
;                  slitw_arc,rp,xunits,yunits,xtitle,ytitle,version, $
;                  PSFRADIUS=psfradius,PSBGINFO=psbginfo,XSBGINFO=xsbginfo,$
;                  WAVEINFO=waveinfo,LINCORMAX=lincormax, CANCEL=cancel
;    
; INPUTS:
;     spectra    - An array [*,3,naps*norders] array of spectra where
;                  array [*,0,0] = wavelengths
;                  array [*,1,0] = flux
;                  array [*,2,0] = error
;     xranges    - An array [2,norders] of column numbers where the
;                  orders are completely on the array
;     fullpath   - The fullpath of the file to be written to
;     aimage     - A string of the name if the Aimage
;     sky        - A string of the sky image
;     flat       - A string of the flat field image
;     naps       - The number of apertures
;     orders     - An array of order numbers
;     hdrinfo    - A structure with FITS keywords and values
;     appos      - An array [naps,norders] of aperture positions
;     apradius   - The aperture radius of the extraction
;     ps         - The plate scale in arcseconds per pixel
;     slith_pix  - The slit length in pixels
;     slith_arc  - The slit length in arcseconds
;     slitw_pix  - The slit width in pixels
;     slitw_arc  - The slit width in arcseconds
;     rp         - Resolving power
;     xunits     - A string giving the units of array[*,0,0]
;     yunits     - A string giving the units of array[*,1,0]
;     xtitle     - A string of the Xtitle to be used for IDL plotting
;     ytitle     - A string of the Ytitle to be used for IDL plotting
;     version    - Spextool version number
;    
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     PSFRADIUS - The radius of the psf in the optimal extraction
;     PSBGINFO  - The background info for standard point source
;                 background definitions [bgstart,bgwidth].
;                 bgstart - The radius in arcseconds at which to start the
;                           background region definition (see mkmask_ps)
;                 bgwidth - The width of the background region in arcseconds
;                           (see mkmask_ps)
;     XSBGINFO  - (Background Regions) Array of background regions 
;                 in arcseconds ([[1,2],[13-15]]).
;     WAVEINFO  - A 3-tag structure.
;                 waveinfo.wavecal = name of the wavecal file
;                 waveinfo.wavetype = 'vacuum' or 'air'
;                 waveinfo.disp = [norders] array of dispersons (um pix-)
;     LINCORMAX - The maxmimum of the linearity correction in DN.
;     CANCEL   - Set on return if there is a problem
;     
; OUTPUTS:
;     Writes a FITS to disk
;     
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     NA
;
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;     2000-09-01 - Written by M. Cushing, Institute for Astronomy, UH
;     2002-10-09 - Added LINEARITY keyword
;     2003-02    - Added optimal extraction output
;     2005-07-04 - Modified to accept the new hdrinfo structure
;     2005-08-04 - Moved the xunits and ynunits parameters and added the
;                  xtitle and ytitle input parameters
;     2005-09-05 - Changed ARCIMAGE keyword to WAVECAL 
;     2006-06-09 - Added appos input
;     2007-07-23 - Added instrument input
;     2008-02-22 - Added rp (resolving power) input
;                - Added the PSBGINFO and XSBGINFO to merge
;                  writespec_xs and writespec_ps
;     2008-03-07 - Added RMS1D and RMS2D keywords.
;     2010-09-05 - Added itot input.
;     2010-10-09 - Added WAVETYPE keyword.
;     2015-01-08 - Added LINCORMAX keyword.
;     2017-11-10 - Modifications for iSHELL
;-
pro mc_writespec,spectra,xranges,fullpath,aimage,sky,flat,naps,orders, $
                 hdrinfo,appos,apradius,ps,slith_pix,slith_arc,slitw_pix, $
                 slitw_arc,rp,xunits,yunits,xtitle,ytitle,steps,version, $
                 PSFRADIUS=psfradius,PSBGINFO=psbginfo,XSBGINFO=xsbginfo,$
                 WAVEINFO=waveinfo,LINCORMAX=lincormax,CANCEL=cancel

  cancel = 0

;  Check parameters

  if n_params() lt 23 then begin
     
     print, 'Syntax - mc_writespec,spectra,xranges,fullpath,aimage,sky,$'
     print, '                      flat,naps,orders,hdrinfo,appos,apradius,$'
     print, '                      ps,slith_pix,slith_arc,slitw_pix,slitw_arc,$'
     print, '                      rp,xunits,yunits,xtitle,ytitle,steps,$'
     print, '                      version,PSFRADIUS=psfradius,$'
     print, '                      PSBGINFO=psbginfo,XSBGINFO=xsbginfo,$'
     print, '                      WAVEINFO=waveinfo,LINCORMAX=lincormax,$'
     print, '                      CANCEL=cancel'
     
     cancel = 1
     return
     
  endif
  cancel = mc_cpar('mc_writespec',spectra,1,'Spectra',[2,3,4,5],[2,3])
  if cancel then return
  cancel = mc_cpar('mc_writespec',xranges,2,'Xranges',[2,3,4,5],[1,2]) 
  if cancel then return
  cancel = mc_cpar('mc_writespec',fullpath,3,'Fullpath',7,0) 
  if cancel then return
  cancel = mc_cpar('mc_writespec',aimage,4,'Aimage',7,0) 
  if cancel then return
  cancel = mc_cpar('mc_writespec',sky,5,'Sky',7,0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',flat,6,'Flat',7,0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',naps,7,'Naps',[2,3,4,5],0) 
  if cancel then return
  cancel = mc_cpar('mc_writespec',orders,8,'Orders',[2,3,4,5],[0,1]) 
  if cancel then return
  cancel = mc_cpar('mc_writespec',hdrinfo,9,'Hdrinfo',8,[0,1])
  if cancel then return
  cancel = mc_cpar('mc_writespec',appos,10,'Appos',[2,3,4,5],[1,2]) 
  if cancel then return
  cancel = mc_cpar('mc_writespec',apradius,11,'Apradius',[2,3,4,5],[0,1]) 
  if cancel then return
  cancel = mc_cpar('mc_writespec',ps,12,'Ps',[2,3,4,5],0) 
  if cancel then return  
  cancel = mc_cpar('mc_writespec',slith_pix,13,'Slith_pix',[2,3,4,5],0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',slith_arc,14,'Slith_arc',[2,3,4,5],0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',slitw_pix,15,'Slitw_pix',[2,3,4,5],0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',slitw_arc,16,'Slitw_arc',[2,3,4,5],0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',rp,17,'Resolving Power',[2,3,4,5],0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',xunits,18,'Xunits',7,0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',yunits,19,'Yunits',7,0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',xtitle,20,'Xtitle',7,0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',ytitle,21,'Ytitle',7,0)
  if cancel then return
  cancel = mc_cpar('mc_writespec',steps,22,'Steps',[1,2],1)
  if cancel then return  
  cancel = mc_cpar('mc_writespec',version,23,'Version',7,0)
  if cancel then return

;  Make stock FITS header.

  fxhmake,hdr,spectra

;  Now add all the keywords from the hdrinfo structure.

  norders  = n_elements(orders)
  ntags    = n_tags(hdrinfo.vals)
  names = tag_names(hdrinfo.vals)

;  The extra loop is because the history has multiple values
  
  l = 0
  for i = 0, ntags-1 do begin
     
     for k = 0,n_elements(hdrinfo.vals.(i))-1 do begin

        if names[i] eq 'HISTORY' then begin

           phistory = (l eq 0) ? (hdrinfo.vals.(i))[k]: $
                      [phistory,(hdrinfo.vals.(i))[k]]
           l++
           continue

        endif
        
        if size((hdrinfo.vals.(i))[k],/TYPE) eq 7 and $
           strlen((hdrinfo.vals.(i))[k]) gt 68 then begin

           fxaddpar,hdr,names[i],(hdrinfo.vals.(i))[k],(hdrinfo.coms.(i))[k]
              
        endif else sxaddpar,hdr,names[i],(hdrinfo.vals.(i))[k], $
                            (hdrinfo.coms.(i))[k]
        
     endfor
     
  endfor

;  Now start adding our keywords
  
  sxaddpar,hdr,'CREPROG','xspextool', ' Creation program'
  sxaddpar,hdr,'VERSION',version, ' Spextool version'
  sxaddpar,hdr,'AMPCOR', steps[0], ' Amplifier correction (bit flag)'
  sxaddpar,hdr,'LINCOR', steps[1], ' Linearity correction (bit flag)'
  sxaddpar,hdr,'FLATED', steps[2], ' Flat fielded (bit flag)'
  sxaddpar,hdr,'FIXBDPX', steps[3], ' Fix bad pixels (bit flag)'
  sxaddpar,hdr,'OPTEXT', steps[4], ' Optimal Extraction (bit flag)'

  sxaddpar,hdr,'AIMAGE',aimage, ' A image'
  lincormax = (keyword_set(LINCORMAX) eq 1) ? lincormax:0
  sxaddpar, hdr, 'LINCRMAX',lincormax,' Linearity correction maximum (DN)'

  fxaddpar,hdr,'SKYORDRK',sky, ' Sky or dark image'
  fxaddpar,hdr,'FLAT',flat, ' Flat field image'

  if keyword_set(WAVEINFO) then begin
     
     sxaddpar,hdr,'WAVECAL',waveinfo.wavecal,' Wavecal file'
     sxaddpar,hdr,'WAVETYPE',waveinfo.wavetype, ' Wavelength type'
     sxaddpar,hdr,'WCTYPE',waveinfo.wctype,' Wavelength calibration type'
     if n_tags(waveinfo) eq 5 then begin

        sxaddpar,hdr,'RECTMETH',waveinfo.rectmeth, $
                 ' Rectification method'        

     endif
     
  endif
  
  if keyword_set(TRACEFILE) then $
     sxaddpar,hdr,'TRACEFIL',tracefile,' Trace file name'
  
  sxaddpar,hdr,'NAPS',fix(naps), ' Number of apertures'
  sxaddpar,hdr,'NORDERS',fix(norders), ' Number of orders'
  fxaddpar,hdr,'ORDERS',strjoin(strcompress(fix(orders),/re),','), $
           ' Order numbers'

  sxaddpar,hdr,'PLTSCALE',ps,FORMAT='(F5.3)',' Plate scale (arcsec pixel-1)'
  sxaddpar,hdr,'SLTH_PIX',slith_pix,FORMAT='(G0.5)', $
           ' Nominal slit height (pixels)'
  sxaddpar,hdr,'SLTH_ARC',slith_arc,FORMAT='(G0.5)', $
           ' Slit height (arcseconds)'
  sxaddpar,hdr,'SLTW_PIX',slitw_pix,FORMAT='(G0.5)', $
           ' Slit width (pixels)'
  sxaddpar,hdr,'SLTW_ARC',slitw_arc,FORMAT='(G0.5)', $
           ' Slit width (arcseconds)'
  sxaddpar,hdr,'RP',long(rp), ' Slit-limited average resolving power'

;  Now add the xranges
  
  for i = 0, norders-1 do begin
     
     name    = 'XROR'+string(orders[i],FORMAT='(i3.3)')
     comment = ' Extraction range for order '+string(orders[i],FORMAT='(i3.3)')
     sxaddpar,hdr,name,strjoin(strtrim(xranges[*,i],2),',',/SINGLE),comment
     
  endfor

;  Add the aperture positions
  
  for i = 0, norders-1 do begin
     
     name = 'APOSO'+string(orders[i],format='(i3.3)')
     comment = ' Aperture positions (arcseconds) for order '+ $
               string(orders[i],FORMAT='(i3.3)')
     sxaddpar,hdr,name,strjoin(strtrim(mc_sigfig(appos[*,i],3),2),','), $
              comment
     
  endfor

;  Add extraction parameters

  psfradius = (keyword_set(PSFRADIUS) eq 1) ? psfradius:0
  sxaddpar,hdr,'PSFRAD',psfradius,' PSF radius (arcseconds)',FORMAT='(f4.2)'
  sxaddpar,hdr,'APRADII',strjoin(apradius,','),' Aperture radii (arcseconds)', $
           FORMAT='(f4.2)'

  if n_elements(PSBGINFO) ne 0 then begin
     
     sxaddpar,hdr,'BGSTART',psbginfo[0], $
              ' Background start radius (arcseconds)',FORMAT='(f4.2)'
     
     sxaddpar,hdr,'BGWIDTH',psbginfo[1],' Background width (arcseconds)', $
              FORMAT='(G0.5)'
     
     sxaddpar,hdr,'BGORDER',psbginfo[2],' Background polynomial fit degree',$
              FORMAT='(I1.1)'
     
  endif
  
  if n_elements(XSBGINFO) ne 0 then begin
     
     sxaddpar,hdr,'BGR',xsbginfo[0],' Background regions (arcseconds)'
     sxaddpar,hdr,'BGORDER',fix(xsbginfo[1]),' Background polynomial fit degree'
     
  endif

;  Add the units
  
  sxaddpar,hdr,'XUNITS',xunits, ' Units of the X axis'
  sxaddpar,hdr,'YUNITS',yunits, ' Units of the Y axis'
  sxaddpar,hdr,'XTITLE',xtitle, ' IDL X title'
  sxaddpar,hdr,'YTITLE',ytitle, ' IDL Y title'

;  Add the dispersion in xunits pixel-1
  
  if keyword_set(WAVEINFO) then begin
     
     for j = 0, norders-1 do begin
        
        name = 'DISPO'+string(orders[j],format='(i3.3)')
        comment = ' Dispersion ('+xunits+' pixel-1) for order '+ $
                  string(orders[j],FORMAT='(i3.3)')
        sxaddpar,hdr,name,waveinfo.disp[j],comment,FORMAT='(G0.5)'
        
     endfor
     
  endif

;  Write the history 

; Write old history

  if n_elements(phistory) then $
     for i = 0,n_elements(phistory)-1 do sxaddhist,phistory[i],hdr
  
  sxaddhist,' ',hdr
  sxaddhist,'######################## Xspextool History ' + $
            '########################',hdr
  sxaddhist,' ',hdr

  history = 'Spextool FITS files contain an array of size ' + $
            '[nwaves,4,norders*naps]. The ith image (array[*,*,i]) ' + $
            'contains the data for a single extraction aperture within ' + $
            'an order such that, lambda=array[*,0,i], flux=array[*,1,i], ' + $
            'uncertainty=array[*,2,i],flag=array[*,3,i].  The zeroth ' + $
            'image (array[*,*,0]) contains the data for the aperture in ' + $
            'the order closest to the bottom of the detector that is ' + $
            'closest to the bottom of the slit (i.e. also closest to the ' + $
            'bottom of the detector).  Moving up the detector, the FITS ' + $
            'array is filled in with subsequent extraction apertures.  ' + $
            'If no orders have been deselected in the extraction process, ' + $
            'the contents of the ith aperture in order j can be found as ' + $
            'follows: lambda=array[*,0,{j-min(orders)}*naps + (i-1)], ' + $
            'flux=array[*,1,{j-min(orders)}*naps + (i-1)], ' + $
            'uncertainty=array[*,2,{j-min(orders)}*naps + (i-1)], ' + $
            'flag=array[*,3,{j-min(orders)}*naps + (i-1)].'

  history = mc_splittext(history,67,CANCEL=cancel)
  if cancel then return
  sxaddhist,history,hdr
    
  writefits,fullpath,spectra,hdr
  

end

