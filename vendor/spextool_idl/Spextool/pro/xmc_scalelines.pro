;+
; NAME:
;     xmc_scalelines
;    
; PURPOSE:
;     Scales the EW of the absorption lines in the Vega spectrum.    
;
; CATEGORY:
;     Widget
;
; CALLING SEQUENCE:
;     xmc_scalelines,std,stdorders,stdmag,stdbmv,wvega,fvega,fcvega,fc2vega,$
;                    kernels,vshift,obj,objorders,objnaps,awave,atrans,$
;                    hlines,hnames,initscale,scales,vrot,PARENT=parent,$
;                    XPUTS=xputs,XTITLE=xtitle,YTITLE=ytitle,CANCEL=cancel
;    
; INPUTS:
;     std        - The standard spectra
;     stdorders  - The order numbers of the standard spectra
;     stdmag     - The V magnitude of the standard
;     stdbmv     - The (B-V) color of the standard
;     wvega      - The wavelength array of the Vega model
;     fvega      - The flux array of the Vega model
;     fcvega     - The continuum array of the Vega model
;     fc2vega    - The fit to the continuum array of the Vega model
;     kernels    - A structure [ntags] of kernels to convolve the Vega model 
;     vshift     - The velocity shift of Vega
;     obj        - The object spectra
;     objorders  - The orders of the object
;     objnaps    - The number of apertures in the object spectra
;     awave      - The atmospheric transmission wavelength array
;     atrans     - The atmopsheric transmission flux array
;     hlines     - An array of hydrogren lines
;     hnames     - A string array of the names of the hydrogen lines
;     initscales - The initial scale factor for the EWs
;     
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     PARENT   - If given, the widget belonging to PARENT is greyed
;                out
;     XPUTS    - An [*,2] array giving the wavelength and transmission
;                of the throughput of the instrument.  If given, the
;                it is multiplied by the atmospheric transmission.
;     XTITLE   - A string of the x-axis title
;     YTITLE   - A string of the y-axis title
;     CANCEL   - Set on return if there is a problem
;     
; OUTPUTS:
;     vshift - The new velocity shift of Vega
;     scales - An array [*,norders] of scale factors for each order
;     
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     xmc_scalelines_state
;
; SIDE EFFECTS:
;     Greys out PARENT
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     The telluric correction spectrum is displayed for each order.
;     The user then adjusts the scale factor for each line until it disappears.
;
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;     2002 - Written by M. Cushing, Institute for Astronomy, UH
;     2005-08-04 - Changed the XUNITS and YUNITS keywords to XTITLE
;                  and YTITLE
;     2005-10-03 - Fixed a bug where you couldn't turn off the
;                  atmosphere and throughputs
;     2018-05-15 - Added the ability to deal with variable R spectra.
;-
;
;==============================================================================
;
;---------------------------- Support Procedures ----------------------------
;
;==============================================================================
;
pro xmc_scalelines_cleanup,base

  widget_control, base, GET_UVALUE = state, /NO_COPY
  if n_elements(state) ne 0 then begin

     ptr_free, state.cpoints
     ptr_free, state.scalespec
     ptr_free, state.tellspec
     ptr_free, stateave
     ptr_free, state.flux
     ptr_free, state.error
     ptr_free, state.spec

  endif
  state = 0B
  
end
;
;==============================================================================
;
pro xmc_scalelines_initcommon,std,stdorders,stdmag,stdbmv,wvega,fvega,fcvega,$
                              fc2vega,kernels,vshift,obj,objorders,objnaps, $
                              awave,atrans,hlines,hnames,initscale,scales,$
                              VARRINFO=varrinfo,XPUTS=xputs,VROT=vrot, $
                              XTITLE=xtitle,YTITLE=ytitle
  
  cleanplot,/SILENT
  
  if n_elements(XPUTS) ne 0 then begin
     
     ixputs = xputs
     doxputs = 1
     
  endif else begin
     
     ixputs = 0
     doxputs = 0
     
  endelse 
  
  xtitle = n_elements(XTITLE) eq 0 ? '':xtitle
  ytitle = n_elements(YTITLE) eq 0 ? '':ytitle
  
;  Create cut region structure
  
  norders = n_elements(stdorders)
  for i = 0,norders-1 do begin
     
     key = string(i)
     value = replicate(!values.f_nan,1)
     cutreg=(i eq 0) ? create_struct(key,ptr_new(value)):$
            create_struct(cutreg,key,ptr_new(value))
     
  endfor
  
  if ~keyword_set(VARRINFO) then varrinfo = 0
  
  common xmc_scalelines_state, state
  
;  Build three structures which will hold important info.
  
  state = {absxrange2:[0.,0.],$
           absyrange2:[0.,0.],$     
           atrans:[[awave],[atrans]],$
           buffer:[0.,0.],$
           cancel:0,$
           cpoints:ptr_new(2),$
           cursor:0,$
           cursormode:'None',$
           cutreg:cutreg,$
           ereg:[!values.f_nan,!values.f_nan],$
           error:ptr_new(2),$
           flux:ptr_new(2),$
           hlines:hlines,$
           hnames:hnames,$
           initscale:initscale,$
           kernels:kernels,$
           keyboard:0L,$
           message:0L,$
           modcpt:-1,$
           norders:n_elements(stdorders),$
           obj:obj,$
           objorders:objorders,$
           objnaps:objnaps,$
           ofcvega:fcvega,$
           ofc2vega:fc2vega,$
           ofvega:fvega,$
           order_dl:0L,$
           owvega:wvega,$
           pixmap1_wid:0L,$
           pixmap2_wid:0L,$
           plot1size:[0.5,0.25]*get_screen_size(),$
           plot1scale:0.,$
           plot2size:[0.5,0.27]*get_screen_size(),$
           plot2scale:0.,$
           plotatmos:1,$
           plotxputs:doxputs,$
           plotwin1:0,$
           plotwin2:0,$
           plotwin1_wid:0L,$
           plotwin2_wid:0L,$
           pscale1:!p,$
           pscale2:!p,$
           reg:[[!values.f_nan,!values.f_nan],$
                [!values.f_nan,!values.f_nan]],$
           scaleatmos_fld:[0L,0L],$
           scalespec:ptr_new(2),$
           scalezero_fld:[0L,0L],$
           slider:0L,$
           spec:ptr_new(2),$
           spectype:'Telluric',$
           std:std,$
           stdidx:0,$
           stdbmv:stdbmv,$
           stdmag:stdmag,$
           stdorders:stdorders,$
           subregscale_fld:[0L,0L],$
           tension:10.,$
           tension_fld:[0L,0L],$
           tellspec:ptr_new(2),$
           varrinfo:varrinfo,$
           vshift:vshift,$
           vshift_fld:[0L,0L],$
           wave:ptr_new(2),$
           xmc_scalelines_base:0L,$
           xmin_fld:[0L,0L],$
           xmax_fld:[0L,0L],$
           xputs:ixputs,$
           xrange2:[0.,0.],$
           xscale1:!x,$
           xscale2:!x,$
           xtitle:xtitle,$           
           ymin_fld:[0L,0L],$
           ymax_fld:[0L,0L],$
           yrange1:[0.,0.],$
           yrange2:[0.,0.],$
           yscale1:!y,$
           yscale2:!y,$
           ytitle:[ytitle,'!5Arbitrary Flux']}           
  
  state.plot1scale = float(state.plot1size[1])/ $
                     (state.plot1size[1]+state.plot2size[1])
  state.plot2scale = float(state.plot2size[1])/ $
                     (state.plot1size[1]+state.plot2size[1])
  
;  Load up the initial control points
  
  for i = 0,state.norders-1 do begin
     
     min   = min(state.std[*,0,i],MAX=max,/NAN)
     array = [min,initscale]
     z     = where(state.hlines gt min and state.hlines lt max,count)
     
     if count ne 0 then begin
        
        hlines = state.hlines[z]
        for j = 0,count-1 do begin
           
           array = [[array],[hlines[j],initscale]]
           
        endfor
        
     endif
     array = [[array],[max,initscale]]
     s = sort(array[0,*])
     array = array[*,s]
     
     key = 'Order'+string(i,FORMAT='(i3)')
     *state.cpoints = (i eq 0) ? create_struct(key,array):$
                      create_struct(*state.cpoints,key,array)
     
  endfor
  
end
;
;==============================================================================
;
pro xmc_scalelines_estimatescale

  common xmc_scalelines_state
  widget_control, /HOURGLASS
  
  std_wave = *state.wave
  std_flux = *state.flux
  std_err  = *state.error
  tel_flux = 1./*state.tellspec
  
  z = where(std_wave gt state.ereg[0] and std_wave lt state.ereg[1])
  
;  Get estimates for the peak finding routine
  
  idx = where(state.hlines gt state.ereg[0] and state.hlines lt state.ereg[1], $
              count)
  if count eq 0 then return
  wguess = state.hlines[idx]
    
  tabinv,std_wave,wguess,idx
  tabinv,std_wave,state.ereg[0],idlo
  tabinv,std_wave,state.ereg[1],idhi
  fguess = tel_flux[idx] - 0.5*(tel_flux[idlo] + tel_flux[idhi])
  sguess = (state.ereg[1]-state.ereg[0])/5.
  cguess = mc_polyfit1d(std_wave[z],tel_flux[z],1,/SILENT)
  mguess = (tel_flux[idlo] - tel_flux[idhi])/(std_wave[idlo]-std_wave[idhi])
  
  result = mpfitpeak(std_wave[z],tel_flux[z],gcoeff,NTERMS=5,/LORENTZIAN,$
                     ESTIMATES=[fguess,wguess,sguess,cguess,mguess])
  
  wline = gcoeff[1]
  
  wvega = state.owvega*(1+state.vshift/2.99792458E5)
  
;  Determine the range over which to convolve the Vega model
  
  wmin = min(std_wave,/NAN,MAX=wmax)
  zv   = where(wvega gt wmin and wvega lt wmax,count)
  
  nkern = n_elements(state.kernels.(state.stdidx))
  nadd  = round(nkern/2.)
  
  idx = findgen(count+2*nadd)+(zv[0]-nadd)
  
;  Do the convolution
  
  rvred    = 3.10
  vegabmv  = 0.00
  vegamag  = 0.03
  magscale = 10.0^(-0.4*(state.stdmag-vegamag))
  ebmv     = (state.stdbmv - vegabmv) > 0.0 
  
; to prevent de-reddening the spectrum
  
  avred    = rvred*ebmv
  redfact  = 10.0^(0.4*avred)
  magfact  = replicate(redfact*magscale,n_elements(std_wave))
  
  nfvconv = convol((state.ofvega[idx]/state.ofcvega[idx]-1.0), $
                   state.kernels.(state.stdidx))
  
  linterp,wvega[idx],nfvconv,std_wave,rnfvconv
  linterp,wvega,state.ofcvega,std_wave,rfcvega
  
;  Redden the convolved Vega model
  
  mc_redden,std_wave,rfcvega,ebmv,rrfcvega
  
; Determine the scale factor
  
  sgwid = 5
;sgdeg = 2
  s = (poly_smooth(std_flux,sgwid)/$
       (magfact*poly(std_wave,gcoeff[3:4])*rrfcvega))- 1.0
  s = temporary(s)/rnfvconv
  
  z = where((*state.cpoints).(state.stdidx)[0,*] gt state.ereg[0] and $
            (*state.cpoints).(state.stdidx)[0,*] lt state.ereg[1],count)
  if count eq 0 then return
  wline = (*state.cpoints).(state.stdidx)[0,z]
  
  tabinv,std_wave,wline,idx
  (*state.cpoints).(state.stdidx)[1,z] = s[idx]
  
end
;
;==============================================================================
;
pro xmc_scalelines_getminmax
  
  common xmc_scalelines_state
  
  state.xrange2 = [min((*state.spec)[*,0],MAX=max),max]
  state.yrange2 = [min((*state.spec)[*,1],MAX=max,/NAN),max]
  
  state.absxrange2 = state.xrange2
  state.absyrange2 = state.yrange2
  
end
;
;===============================================================================
;
pro xmc_scalelines_makescale
  
  common xmc_scalelines_state
  
  if n_elements((*state.cpoints).(state.stdidx)[0,*]) eq 2 then begin
     
     scalespec = replicate(1.0,n_elements(*stateave))
     
  endif else begin
     
     scalespec = spline((*state.cpoints).(state.stdidx)[0,*],$
                        (*state.cpoints).(state.stdidx)[1,*],$
                        *state.wave,state.tension)
     
  endelse
  
  *state.scalespec = scalespec
  
end
;
;==============================================================================
;
pro xmc_scalelines_makespec

  common xmc_scalelines_state
  
  case state.spectype of
     
     'Telluric': begin
        
        wave = *state.wave
        spec = *state.tellspec
        *state.spec = [[wave],[1./spec]]
        
     end
     
     'Object': begin
        
        idx = mc_nantrim(state.obj[*,0,state.objnaps*state.stdidx],2)
        wave = state.obj[idx,0,state.objnaps*state.stdidx]
        spec = state.obj[idx,1,state.objnaps*state.stdidx]
        
        mc_interpspec,*state.wave,*state.tellspec,wave,tellspec,CANCEL=cancel
        if cancel then return
        
        spec = temporary(spec)*tellspec
        *state.spec = [[wave],[spec]]
        
     end
     
  endcase
  
end
;
;==============================================================================
;
pro xmc_scalelines_maketelluric

  common xmc_scalelines_state

  if size(state.varrinfo,/TYPE) eq 8 then varrinfo = state.varrinfo

  mc_mktellspec, *state.wave,*state.flux,*state.error,state.stdmag, $
                 state.stdbmv,state.kernels.(state.stdidx),state.owvega, $
                 state.ofvega,state.ofcvega,state.ofc2vega,state.vshift, $
                 tellcor,VARRINFO=varrinfo,SCALES=*state.scalespec,CANCEL=cancel
  if cancel then return
  
;  Perform interpolations if necessary
  
  ndat = n_elements(*state.cutreg.(state.stdidx))
  
  if ndat ne 1 then begin
     
     nreg = ndat/2
     for i = 0, nreg-1 do begin
        
        xrange = reform((*state.cutreg.(state.stdidx))[(i*2):(i*2+1)])
        tabinv,*state.wave,xrange,idx
        idx = round(idx)
        
        x = [(*state.wave)[idx[0]],(*state.wave)[idx[1]]]
        
        y = [tellcor[idx[0]],tellcor[idx[1]]]
        coeff = mc_polyfit1d(x,y,1,/SILENT)
        
        tellcor[idx[0]:idx[1]]=poly((*state.wave)[idx[0]:idx[1]],coeff)
        
     endfor
         
  endif
  
  *state.tellspec = tellcor
  
end
;
;==============================================================================
;
pro xmc_scalelines_plotscale

  common xmc_scalelines_state
  
  plotsym,8,1.3,/FILL
  if state.modcpt eq -1 then begin
     
     min = min(*state.scalespec,max=max)
     del = (max-min)*0.3
     if del lt 1e-2 then del = 0.5
     state.yrange1 = [min-del,max+del]
     
  endif
  
  plot,(*state.spec)[*,0],*state.scalespec,/XSTY,/YSTY,$
       YRANGE=state.yrange1,XRANGE=state.xrange2,YTITLE='Scale Factor',$
       XTITLE=state.xtitle,PSYM=10,CHARSIZE=1.5
  
  ncpts = n_elements((*state.cpoints).(state.stdidx)[0,*])
  
  for i = 0, ncpts-1 do begin
     
     if (*state.cpoints).(state.stdidx)[0,i] ge state.xrange2[0] and $
        (*state.cpoints).(state.stdidx)[0,i] le state.xrange2[1] then $
           plots,[(*state.cpoints).(state.stdidx)[0,i],$
                  (*state.cpoints).(state.stdidx)[1,i]],PSYM=8,COLOR=3
     
  endfor

  state.pscale1 = !p
  state.xscale1 = !x
  state.yscale1 = !y

end
;
;==============================================================================
;
pro xmc_scalelines_plotspec

  common xmc_scalelines_state
  
  scale = mc_cfld(state.scaleatmos_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  z = where(state.atrans[*,0] lt state.absxrange2[1] and $
            state.atrans[*,0] gt state.absxrange2[0],count)
  
  wtrans = state.atrans[z,0] 
  spec   = state.atrans[z,1] 
  spec   = (temporary(spec)-1.0)*scale+1.0
  
  if state.plotatmos then begin
     
     z = where( wtrans lt state.xrange2[1] and wtrans gt state.xrange2[0],count)
     
     if count ne 0 then begin
        
        plot, wtrans,spec,COLOR=5,YRANGE=[0,1],YSTYLE=5,XSTYLE=5,$
              XRANGE=state.xrange2,PSYM=10,CHARSIZE=1.5
        ticks = string(findgen(11)*.1,FORMAT='(f3.1)')
        axis,YAXIS=1,YTICKS=10,YTICKNAME=ticks,YMINOR=1,COLOR=5
        
     endif
     ystyle = 9
     noerase = 1
     
  endif else begin
     
     ystyle=1
     noerase = 0
     
  endelse
  
  case state.spectype of
     
     'Telluric': ytitle = state.ytitle[0]
     
     'Object': ytitle = state.ytitle[1]
     
  endcase
  
  plot,(*state.spec)[*,0],(*state.spec)[*,1],XTITLE=state.xtitle, $
       YTITLE=ytitle,/XSTY,YSTYLE=ystyle,YRANGE=state.yrange2, $
       XRANGE=state.xrange2,NOERASE=noerase,PSYM=10,CHARSIZE=1.5
  
;  Label H lines
  
  z = where(state.hlines lt state.xrange2[1] and $
            state.hlines gt state.xrange2[0],count)
  
  for i =0, count-1 do begin
     
     tabinv,(*state.spec)[*,0],state.hlines[z[i]],idx
     
     linenorm = convert_coord(state.hlines[z[i]],(*state.spec)[idx,1], $
                              /DATA,/TO_NORM)

     plotnorm = convert_coord(!x.crange[0],!y.crange[1],/DATA,/TO_NORM)

     ytop = (linenorm[1]+0.2) < (plotnorm[1]-0.03)

     plots,[linenorm[0],linenorm[0]],[linenorm[1],ytop],LINESTYLE=1,COLOR=3, $
           THICK=2,/NORM
     
     xyouts, linenorm[0],ytop+0.005,state.hnames[z[i]]+'!X',ORIENTATION=90, $
             /NORM,COLOR=3
     
  endfor
  
  state.pscale2 = !p
  state.xscale2 = !x
  state.yscale2 = !y
  
end
;
;==============================================================================
;
pro xmc_scalelines_plotupdate1

  common xmc_scalelines_state
  
  wset, state.pixmap1_wid
  erase
  xmc_scalelines_plotscale
  
  wset, state.plotwin1_wid
  device, COPY=[0,0,state.plot1size[0],state.plot1size[1],0,0,state.pixmap1_wid]

end
;
;==============================================================================
;
pro xmc_scalelines_plotupdate2

  common xmc_scalelines_state
  
  wset, state.pixmap2_wid
  erase
  xmc_scalelines_plotspec 
  
  wset, state.plotwin2_wid
  device, COPY=[0,0,state.plot2size[0],state.plot2size[1],0,0,state.pixmap2_wid]

end
;
;==============================================================================
;
pro xmc_scalelines_selectspec

  common xmc_scalelines_state
  
  idx = mc_nantrim(state.std[*,0,state.stdidx],2)
  *state.wave = reform(state.std[idx,0,state.stdidx])
  *state.flux = reform(state.std[idx,1,state.stdidx])
  *state.error = reform(state.std[idx,2,state.stdidx])

end
;
;==============================================================================
;
pro xmc_scalelines_setminmax

  common xmc_scalelines_state
  
  widget_control, state.xmin_fld[1],SET_VALUE=strtrim(state.xrange2[0],2)
  widget_control, state.xmax_fld[1],SET_VALUE=strtrim(state.xrange2[1],2)
  widget_control, state.ymin_fld[1],SET_VALUE=strtrim(state.yrange2[0],2)
  widget_control, state.ymax_fld[1],SET_VALUE=strtrim(state.yrange2[1],2)

end
;
;==============================================================================
;
pro xmc_scalelines_undocut

  common xmc_scalelines_state
  
  ndat = n_elements(*state.cutreg.(state.stdidx))
  if ndat ne 1 then begin
     
     if ndat eq 2 then *state.cutreg.(state.stdidx) = !values.f_nan
     if ndat gt 2 then *state.cutreg.(state.stdidx) = $
        (*state.cutreg.(state.stdidx))[0:ndat-3]
     
  endif
  
  xmc_scalelines_maketelluric
  xmc_scalelines_makespec
  xmc_scalelines_plotupdate2   
  
end
;
;==============================================================================
;
pro xmc_scalelines_whichpoint,idx

  common xmc_scalelines_state
  
  idx = -1
  del = state.xrange2[1]-state.xrange2[0]
  ncpts = n_elements((*state.cpoints).(state.stdidx)[0,*])
  
  for i = 0,ncpts-1 do begin
     
     if state.reg[0,0] gt (*state.cpoints).(state.stdidx)[0,i]-$
        (del*0.005) and $
        state.reg[0,0] lt (*state.cpoints).(state.stdidx)[0,i]+$
        (del*0.005) then $
           idx = i
     
  endfor
  
end
;
;==============================================================================
;
pro xmc_scalelines_zoom,IN=in,OUT=out

  common xmc_scalelines_state
  
  delabsx = state.absxrange2[1]-state.absxrange2[0]
  delx    = state.xrange2[1]-state.xrange2[0]
  
  delabsy = state.absyrange2[1]-state.absyrange2[0]
  dely    = state.yrange2[1]-state.yrange2[0]
  
  xcen = state.xrange2[0]+delx/2.
  ycen = state.yrange2[0]+dely/2.
  
  case state.cursormode of 
     
     'XZoom': begin
        
        z = alog10(delabsx/delx)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsx/2.^z/2.
        state.xrange2 = [xcen-hwin,xcen+hwin]
        xmc_scalelines_plotupdate1
        xmc_scalelines_plotupdate2
        xmc_scalelines_setminmax
        
     end
     
     'YZoom': begin
        
        z = alog10(delabsy/dely)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsy/2.^z/2.
        state.yrange2 = [ycen-hwin,ycen+hwin]
        xmc_scalelines_plotupdate2
        xmc_scalelines_setminmax
        
     end
     
     'Zoom': begin
        
        z = alog10(delabsx/delx)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsx/2.^z/2.
        state.xrange2 = [xcen-hwin,xcen+hwin]
        
        z = alog10(delabsy/dely)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsy/2.^z/2.
        state.yrange2 = [ycen-hwin,ycen+hwin]
        
        xmc_scalelines_plotupdate1
        xmc_scalelines_plotupdate2
        xmc_scalelines_setminmax
        
     end
     
     else:
     
  endcase
  
end
;  
;===============================================================================
;
;-----------------------------Event Handlers----------------------------------
;
;===============================================================================
;
pro xmc_scalelines_event,event

  common xmc_scalelines_state
  widget_control, event.id,  GET_UVALUE = uvalue
  widget_control, /HOURGLASS
  
  case uvalue of
     
     'Accept': widget_control, event.top, /DESTROY
     
     'Atmosphere': begin
        
        state.plotatmos = event.select
        xmc_scalelines_plotupdate2
        
     end
     
     'Cancel': begin
        
        state.cancel = 1
        widget_control, event.top, /DESTROY
        
     end
     
     'Keyboard': begin
        
        case strtrim(event.ch,2) of 
           
           'c': begin
              
;                if state.cursormode eq 'Select Region' then begin
;                 
;                    z = where(finite(state.scalereg) eq 1,count)
;                    if count eq 0 then begin
;
;                        xmc_scalelines_modoffsets,$
;                          (*state.scalespec)[state.scalereg[0,0]-1]
;                        state.scalereg[*] = !values.f_nan
;                        widget_control, state.regionscale_base, MAP=0
;                        xmc_scalelines_makescale
;                        xmc_scalelines_maketelluric
;                        xmc_scalelines_makespec
;                        
;                    endif else state.scalereg[*] = !values.f_nan
;
;                endif
              
              state.cursormode = 'None'
              state.reg = !values.f_nan
              xmc_scalelines_plotupdate1
              xmc_scalelines_plotupdate2
              
              
           end

           'e': begin
              
              state.cursormode = 'Estimate Scale'
              state.ereg = !values.f_nan
              
           end
           
           'f': begin
              
              state.cursormode='Fix'
              state.reg = !values.f_nan
              
              
           end
           
           'i': xmc_scalelines_zoom,/IN
           
           'o': xmc_scalelines_zoom,/OUT
           
           'w': begin
              
              state.xrange2 = state.absxrange2
              state.yrange2 = state.absyrange2
              xmc_scalelines_setminmax
              xmc_scalelines_plotupdate1
              xmc_scalelines_plotupdate2
              
           end
           
           'u': xmc_scalelines_undocut
           
           'x': begin 
              
              state.cursormode = 'XZoom'
              state.reg = !values.f_nan
              
           end
           
           'y': begin 
              
              state.cursormode = 'YZoom'
              state.reg = !values.f_nan
              
           end
           
           'z': begin
              
              state.cursormode = 'Zoom'
              state.reg = !values.f_nan
              
           end
           
           else:
           
        endcase
        
     end
     
     'Order': begin
        
        state.stdidx = event.index
        xmc_scalelines_selectspec
        xmc_scalelines_makescale
        xmc_scalelines_maketelluric
        xmc_scalelines_makespec
        xmc_scalelines_getminmax
        xmc_scalelines_setminmax
        xmc_scalelines_plotupdate1
        xmc_scalelines_plotupdate2        
        
     end
     
     'Reset Control Points': begin
        
        (*state.cpoints).(state.stdidx)[1,*] = state.initscale
        xmc_scalelines_makescale
        xmc_scalelines_maketelluric
        xmc_scalelines_makespec
        xmc_scalelines_plotupdate1
        xmc_scalelines_plotupdate2                
        
     end
     
     'Scale Atmosphere': xmc_scalelines_plotupdate2
     
     'Spectrum Type': begin
        
        state.spectype = event.value
        xmc_scalelines_makespec
        xrange = state.xrange2
        xmc_scalelines_getminmax
        state.xrange2 = xrange
        z = where((*state.spec)[*,0] gt state.xrange2[0] and $
                  (*state.spec)[*,0] lt state.xrange2[1])
        state.yrange2 = [min((*state.spec)[z,1],max=max,/NAN),max]
        xmc_scalelines_setminmax
        xmc_scalelines_plotupdate2
        
        
     end
     
     'Vrot': begin
        
        val = mc_cfld(state.vrot_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then goto, cont
        state.vrot = val
        xmc_scalelines_maketelluric
        xmc_scalelines_makespec
        xmc_scalelines_plotupdate2       
        
     end
     
     'Vshift': begin
        
        val = mc_cfld(state.vshift_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then goto, cont
        state.vshift = val
        xmc_scalelines_maketelluric
        xmc_scalelines_makespec
        xmc_scalelines_plotupdate2       
        
     end
     
     'Throughputs': begin
        
        state.plotxputs = event.select
        xmc_scalelines_plotupdate2
        
     end
     
     else:
     
  endcase
  
cont: 
  
end
;
;==============================================================================
;
pro xmc_scalelines_minmaxevent,event

  common xmc_scalelines_state
  widget_control, event.id,  GET_UVALUE = uvalue
  
  case uvalue of 
     
     'X Min': begin
        
        xmin = mc_cfld(state.xmin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        xmin2 = mc_crange(xmin,state.xrange2[1],'X Min',/KLT,$
                          WIDGET_ID=state.xmc_scalelines_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control, state.xmin_fld[0],SET_VALUE=state.xrange2[0]
           return
           
        endif else state.xrange2[0] = xmin2
        
     end
     'X Max': begin
        
        xmax = mc_cfld(state.xmax_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        xmax2 = mc_crange(xmax,state.xrange2[0],'X Max',/KGT,$
                          WIDGET_ID=state.xmc_scalelines_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control, state.xmax_fld[0],SET_VALUE=state.xrange2[1]
           return
           
        endif else state.xrange2[1] = xmax2
        
     end
     'Y Min': begin
        
        ymin = mc_cfld(state.ymin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        ymin2 = mc_crange(ymin,state.yrange2[1],'Y Min',/KLT,$
                          WIDGET_ID=state.xmc_scalelines_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control,state.ymin_fld[0],SET_VALUE=state.yrange2[0]
           return
           
        endif else state.yrange2[0] = ymin2
        
     end
     'Y Max': begin
        
        ymax = mc_cfld(state.ymax_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        ymax2 = mc_crange(ymax,state.yrange2[0],'Y Max',/KGT,$
                          WIDGET_ID=state.xmc_scalelines_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control,state.ymax_fld[0],SET_VALUE=state.yrange2[1]
           return
           
        endif else state.yrange2[1] = ymax2
        
     end
     
  endcase
  
  xmc_scalelines_plotupdate1
  xmc_scalelines_plotupdate2
  
end
;
;===============================================================================
;
pro xmc_scalelines_plotwin1event,event

  common xmc_scalelines_state
  widget_control, event.id,  GET_UVALUE = uvalue
  
;  Check to see if it is a TRACKING event.
  
  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin
     
     if event.enter eq 0 then begin
        
        widget_control, state.keyboard, SENSITIVE=0
        wset, state.plotwin1_wid
        device, COPY=[0,0,state.plot1size[0],state.plot1size[1],0,0,$
                      state.pixmap1_wid]
        wset, state.plotwin2_wid
        device, COPY=[0,0,state.plot2size[0],state.plot2size[1],0,0,$
                      state.pixmap2_wid]
        
     endif
     goto, cont
     
  endif else widget_control, state.keyboard, /INPUT_FOCUS, /SENSITIVE
  
  wset, state.plotwin1_wid
  
;  Load up plot scales.
  
  !p = state.pscale1
  !x = state.xscale1
  !y = state.yscale1
  
;  Determine the wavelength and flux values for the event.
  
  x  = event.x/float(state.plot1size[0])  
  y  = event.y/float(state.plot1size[1])
  xy = convert_coord(x,y,/NORMAL,/TO_DATA)
  
  case event.type of 
     
     0: begin
        
        state.reg[*,0] = xy[0:1]
        xmc_scalelines_whichpoint,idx
        state.modcpt = idx
        
     end
     
     1: begin
        
        if state.modcpt ne -1 then begin
           
           widget_control, /HOURGLASS
           tmp = state.modcpt
           state.modcpt = -1
           xmc_scalelines_maketelluric
           xmc_scalelines_makespec
           xmc_scalelines_plotupdate1
           xmc_scalelines_plotupdate2
           
        endif
     end
     
     2: begin
        
        if state.modcpt ne -1 then begin
           
           (*state.cpoints).(state.stdidx)[1,state.modcpt] = xy[1]
           xmc_scalelines_makescale
           xmc_scalelines_plotupdate1
           
        endif
        
     end
     
  endcase
  
;  Copy the pixmap and draw the cross hair.
  
  wset, state.plotwin1_wid
  device, copy=[0,0,state.plot1size[0],state.plot1size[1],0,0,$
                state.pixmap1_wid]
  
  wset, state.plotwin2_wid
  device, copy=[0,0,state.plot2size[0],state.plot2size[1],0,0,$
                state.pixmap2_wid]
  
  wset, state.plotwin1_wid
  plots, [event.x,event.x],[0,state.plot1size[1]],COLOR=2,/DEVICE
  plots, [0,state.plot1size[0]],[event.y,event.y],COLOR=2,/DEVICE
  
  
  tabinv, (*state.spec)[*,0],xy[0],idx
  idx = round(idx)
  label = 'Cursor X: '+strtrim(xy[0],2)+', Y: '+strtrim(xy[1],2)
  label = label+'   Scale X: '+strtrim((*state.spec)[idx,0],2)+$
          ', Scale Y: '+strtrim((*state.scalespec)[idx],2)
  widget_control,state.message,SET_VALUE=label
  
cont:

end
;
;===============================================================================
;
pro xmc_scalelines_plotwin2event,event
  
  common xmc_scalelines_state
  widget_control, event.id,  GET_UVALUE = uvalue
  
;  Check to see if it is a TRACKING event.
  
  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin
     
     if event.enter eq 0 then begin
        
        widget_control, state.keyboard, SENSITIVE=0
        wset, state.plotwin1_wid
        device, COPY=[0,0,state.plot1size[0],state.plot1size[1],0,0,$
                      state.pixmap1_wid]
        wset, state.plotwin2_wid
        device, COPY=[0,0,state.plot2size[0],state.plot2size[1],0,0,$
                      state.pixmap2_wid]
        
     endif
     goto, cont
     
  endif else widget_control, state.keyboard, /INPUT_FOCUS, /SENSITIVE
  
  wset, state.plotwin2_wid
  
;  Load up plot scales.
  
  !p = state.pscale2
  !x = state.xscale2
  !y = state.yscale2
  
;  Determine the wavelength and flux values for the event.
  
  x  = event.x/float(state.plot2size[0])  
  y  = event.y/float(state.plot2size[1])
  xy = convert_coord(x,y,/NORMAL,/TO_DATA)
  
  if event.type eq 1 then begin
     
     case state.cursormode of 
        
        'Estimate Scale': begin
           
           z = where(finite(state.ereg) eq 1,count)
           if count eq 0 then begin
              
              state.ereg[0] = xy[0]
              wset, state.pixmap2_wid
              plots, [event.x,event.x],[0,state.plot2size[1]],COLOR=7,$
                     /DEVICE,LINESTYLE=2
              wset, state.plotwin2_wid
              device, COPY=[0,0,state.plot2size[0],state.plot2size[1],0,$
                            0,state.pixmap2_wid]
              
           endif else begin
              
              state.ereg[1] = xy[0]
              xmc_scalelines_estimatescale
              xmc_scalelines_makescale
              xmc_scalelines_maketelluric
              xmc_scalelines_makespec
              xmc_scalelines_plotupdate1
              xmc_scalelines_plotupdate2
              state.cursormode = 'None'
              
           endelse
           
        end
        
        'Fix': begin
           
           z = where(finite(state.reg) eq 1,count)
           if count eq 0 then begin
              
              state.reg[*,0] = xy[0:1] 
              
              wset, state.plotwin2_wid
              plots,[xy[0],xy[0]],(state.yscale2).crange,LINESTYLE=2,COLOR=7
              wset, state.pixmap2_wid
              plots,[xy[0],xy[0]],(state.yscale2).crange,LINESTYLE=2,COLOR=7
              
           endif else begin
              
              state.reg[*,1] = xy[0:1]
              
              ndat = n_elements(*state.cutreg.(state.stdidx))
              xx = reform(state.reg[0,*])
              xx = xx[sort(xx)]
              
              if ndat eq 1 then *state.cutreg.(state.stdidx) = xx $
              else *state.cutreg.(state.stdidx) = $
                 [*state.cutreg.(state.stdidx),xx]
              
              state.reg = !values.f_nan
              xmc_scalelines_maketelluric
              xmc_scalelines_makespec
              xmc_scalelines_plotupdate2     
              state.cursormode = 'None'
              
           endelse
           
        end

        'XZoom': begin
           
           z = where(finite(state.reg) eq 1,count)
           if count eq 0 then begin
              
              state.reg[*,0] = xy[0:1]
              wset, state.pixmap2_wid
              plots, [event.x,event.x],[0,state.plot2size[1]],COLOR=2,$
                     /DEVICE,LINESTYLE=2
              wset, state.plotwin2_wid
              device, copy=[0,0,state.plot2size[0],state.plot2size[1],0,$
                            0,state.pixmap2_wid]
              
           endif else begin
              
              state.reg[*,1] = xy[0:1]                
              state.xrange2 = [min(state.reg[0,*],MAX=m),m]
              state.cursormode = 'None'
              state.reg = !values.f_nan
              xmc_scalelines_plotupdate1
              xmc_scalelines_plotupdate2
              xmc_scalelines_setminmax
              
           endelse
           
        end
        
        'YZoom': begin
           
           z = where(finite(state.reg) eq 1,count)
           if count eq 0 then begin
              
              state.reg[*,0] = xy[0:1]
              wset, state.pixmap2_wid
              plots, [0,state.plot2size[0]],[event.y,event.y],COLOR=2,$
                     /DEVICE,LINESTYLE=2
              
              wset, state.plotwin2_wid
              device, COPY=[0,0,state.plot2size[0],state.plot2size[1],$
                            0,0,state.pixmap2_wid]
              
           endif else begin
              
              state.reg[*,1] = xy[0:1]
              state.yrange2 = [min(state.reg[1,*],MAX=m),m]
              state.cursormode = 'None'
              state.reg = !values.f_nan
              xmc_scalelines_plotupdate1
              xmc_scalelines_plotupdate2
              xmc_scalelines_setminmax
              
           endelse
           
        end
        
        'Zoom': begin
           
           z = where(finite(state.reg) eq 1,count)
           if count eq 0 then state.reg[*,0] = xy[0:1] else begin
              
              state.reg[*,1] = xy[0:1]
              state.xrange2 = [min(state.reg[0,*],MAX=max),max]
              state.yrange2 = [min(state.reg[1,*],MAX=max),max]
              state.cursormode = 'None'
              state.reg = !values.f_nan
              xmc_scalelines_plotupdate1
              xmc_scalelines_plotupdate2
              xmc_scalelines_setminmax
              
           endelse
           
        end
        
        else:
        
     endcase
     
  endif
  
;  Copy the pixmap and draw the cross hair.
  
  wset, state.plotwin1_wid
  device, COPY=[0,0,state.plot1size[0],state.plot1size[1],0,0,state.pixmap1_wid]
  
  wset, state.plotwin2_wid
  device, COPY=[0,0,state.plot2size[0],state.plot2size[1],0,0,state.pixmap2_wid]
  
  case state.cursormode of 
     
     'XZoom': begin
        
        wset, state.plotwin2_wid
        plots, [event.x,event.x],[0,state.plot2size[1]],color=2,/DEVICE
        wset, state.plotwin1_wid
        plots, [event.x,event.x],[0,state.plot1size[1]],COLOR=2,/DEVICE
        
     end
     
     'YZoom': plots, [0,state.plot2size[0]],[event.y,event.y],COLOR=2,/DEVICE
     
     'Zoom': begin
        
        plots, [event.x,event.x],[0,state.plot2size[1]],COLOR=2,/DEVICE
        plots, [0,state.plot2size[0]],[event.y,event.y],COLOR=2,/DEVICE
        xy = convert_coord(event.x,event.y,/DEVICE,/TO_DATA)
        plots,[state.reg[0,0],state.reg[0,0]],[state.reg[1,0],xy[1]],$
              LINESTYLE=2,COLOR=2
        plots, [state.reg[0,0],xy[0]],[state.reg[1,0],state.reg[1,0]],$
               LINESTYLE=2,COLOR=2
        
     end
     
     else: begin
        
        wset, state.plotwin2_wid
        plots, [event.x,event.x],[0,state.plot2size[1]],COLOR=2,/DEVICE
        plots, [0,state.plot2size[0]],[event.y,event.y],COLOR=2,/DEVICE
        wset, state.plotwin1_wid
        plots, [event.x,event.x],[0,state.plot1size[1]],COLOR=2,/DEVICE
        
     end
     
  endcase
  
;  Update cursor position
  
  label = 'Cursor X: '+strtrim(xy[0],2)+', Y: '+strtrim(xy[1],2)
  widget_control,state.message,SET_VALUE=label

cont:
end
;
;===============================================================================
;
pro xmc_scalelines_resizeevent, event
  
  common xmc_scalelines_state
  
  widget_control, state.xmc_scalelines_base, TLB_GET_SIZE = size
  
;  Get new plot sizes

  state.plot1size[0]=size[0]-state.buffer[0]
  state.plot1size[1]=(size[1]-state.buffer[1])*state.plot1scale
  
  state.plot2size[0]=size[0]-state.buffer[0]
  state.plot2size[1]=(size[1]-state.buffer[1])*state.plot2scale

;  Resize windows
  
  widget_control, state.plotwin1, UPDATE=0
  widget_control, state.plotwin2, UPDATE=0

  widget_control, state.plotwin1, DRAW_XSIZE=state.plot1size[0]
  widget_control, state.plotwin1, DRAW_YSIZE=state.plot1size[1]

  widget_control, state.plotwin2, DRAW_XSIZE=state.plot2size[0]
  widget_control, state.plotwin2, DRAW_YSIZE=state.plot2size[1]

  widget_control, state.plotwin1, UPDATE=1
  widget_control, state.plotwin2, UPDATE=1
  
;  Redo pixel maps

  wdelete,state.pixmap1_wid
  window, /FREE, /PIXMAP,XSIZE=state.plot1size[0],YSIZE=state.plot1size[1]
  state.pixmap1_wid = !d.window
  
  wset, state.plotwin1_wid
  device, COPY=[0,0,state.plot1size[0],state.plot1size[1],0,0,$
                state.pixmap1_wid]

  wdelete,state.pixmap2_wid
  window, /FREE, /PIXMAP,XSIZE=state.plot2size[0],YSIZE=state.plot2size[1]
  state.pixmap2_wid = !d.window
  

  xmc_scalelines_plotupdate1
  xmc_scalelines_plotupdate2

end
;
;==============================================================================
;
;---------------------------------Main Program--------------------------------
;
;==============================================================================
;
pro xmc_scalelines,std,stdorders,stdmag,stdbmv,wvega,fvega,fcvega,fc2vega,$
                   kernels,vshift,obj,objorders,objnaps,awave,atrans,hlines,$
                   hnames,initscale,scales,cutreg,PARENT=parent,XPUTS=xputs,$
                   VARRINFO=varrinfo,XTITLE=xtitle,YTITLE=ytitle,CANCEL=cancel

 common xmc_scalelines_state

 mc_mkct

 if not xregistered('xmc_scalelines') then begin
    
    xmc_scalelines_initcommon,std,stdorders,stdmag,stdbmv,wvega,fvega,fcvega,$
                              fc2vega,kernels,vshift,obj,objorders,objnaps, $
                              awave,atrans,hlines,hnames,initscale,scales, $
                              VARRINFO=varrinfo,XPUTS=xputs,XTITLE=xtitle, $
                              YTITLE=ytitle
    
    if n_elements(PARENT) ne 0 then widget_control, parent, SENSITIVE=0
    
;  Now build the mask for whether there are H lines within a given
;  order
    
    mask = bytarr(state.norders)
    for i = 0,state.norders-1 do begin
       
       min = min(std[*,0,i],MAX=max,/NAN)
       z = where(state.hlines gt min and state.hlines lt max,cnt)
       if cnt ne 0 then mask[i] = 1
       
    endfor
    
;  Build the widget.
    
    mc_getfonts,buttonfont,textfont
    
    state.xmc_scalelines_base = widget_base(TITLE='Xmc_Scalelines', $
                                            /COLUMN,$
                                            /TLB_SIZE_EVENTS)
    
       button = widget_button(state.xmc_scalelines_base,$
                              FONT=buttonfont,$
                              EVENT_PRO='xmc_scalelines_event',$
                              VALUE='Cancel',$
                              UVALUE='Cancel')
       
       state.message = widget_text(state.xmc_scalelines_base, $
                                   VALUE='',$
                                   YSIZE=1)
       
       state.keyboard = widget_text(state.xmc_scalelines_base, $
                                    /ALL_EVENTS, $
                                    SCR_XSIZE=1, $
                                    SCR_YSIZE=1, $
                                    UVALUE='Keyboard', $
                                    EVENT_PRO='xmc_scalelines_event',$
                                    VALUE= '')

       row = widget_base(state.xmc_scalelines_base,$
                         EVENT_PRO='xmc_scalelines_event',$
                         /ROW,$
                         FRAME=2,$
                         /BASE_ALIGN_CENTER)
       
          v = coyote_field2(row,$
                            LABELFONT=buttonfont,$
                            FIELDFONT=textfont,$
                            TITLE='Vshift (km/s) :',$
                            UVALUE='Vshift',$
                            VALUE=strtrim(state.vshift,2),$
                            XSIZE=8,$
                            EVENT_PRO='xmc_scalelines_event',$
                            /CR_ONLY,$
                            TEXTID=textid)
          state.vshift_fld = [v,textid]
          
          button = widget_button(row,$
                                 VALUE='Reset Control Points',$
                                 FONT=buttonfont,$
                                 UVALUE='Reset Control Points')    
          
          
          
       state.plotwin1 = widget_draw(state.xmc_scalelines_base,$
                                    XSIZE=state.plot1size[0],$
                                    YSIZE=state.plot1size[1],$
                                    /TRACKING_EVENTS,$
                                    /BUTTON_EVENTS,$
                                    /MOTION_EVENTS,$
                                    EVENT_PRO='xmc_scalelines_plotwin1event',$
                                    UVALUE='Plot Window 1')
       
       
       row = widget_base(state.xmc_scalelines_base,$
                         EVENT_PRO='xmc_scalelines_event',$
                         FRAME=2,$
                         /ROW,$
                         /BASE_ALIGN_CENTER)

          bg = cw_bgroup(row,$
                         FONT=buttonfont,$
                         ['Telluric','Object'],$
                         /ROW,$
                         /RETURN_NAME,$
                         /NO_RELEASE,$
                         /EXCLUSIVE,$
                         LABEL_LEFT='Spectrum:',$
                         UVALUE='Spectrum Type',$
                         SET_VALUE=0)
          
          vals = string(state.stdorders,FORMAT='(I3)')
          z = where(mask eq 1,cnt)
          if cnt ne 0 then vals[z] = vals[z]+'*'
          
          state.order_dl = widget_droplist(row,$
                                           FONT=buttonfont,$
                                           TITLE='Order:',$
                                           VALUE=vals,$
                                           UVALUE='Order')
          
          atmos_bg = cw_bgroup(row,$
                               ['Atmosphere'],$
                               FONT=buttonfont,$
                               UVALUE='Atmosphere',$
                               SET_VALUE=[1],$
                               /NONEXCLUSIVE)
                 
          fld  = coyote_field2(row,$
                               LABELFONT=buttonfont,$
                               FIELDFONT=textfont,$
                               TITLE='Scale:',$
                               UVALUE='Scale Atmosphere',$
                               XSIZE=4,$
                               VALUE=1.0,$
                               EVENT_PRO='xmc_scalelines_event',$
                               /CR_ONLY,$
                               TEXTID=textid)
          state.scaleatmos_fld = [fld,textid]
       
          if n_elements(XPUTS) ne 0 then begin
             
             xput_bg = cw_bgroup(row,$
                                 ['Throughputs'],$
                                 FONT=buttonfont,$
                                 UVALUE='Throughputs',$
                                 SET_VALUE=[1],$
                                 /NONEXCLUSIVE)
             
          endif

       state.plotwin2 = widget_draw(state.xmc_scalelines_base,$
                                    XSIZE=state.plot2size[0],$
                                    YSIZE=state.plot2size[1],$
                                    /TRACKING_EVENTS,$
                                    /BUTTON_EVENTS,$
                                    /MOTION_EVENTS,$
                                    EVENT_PRO='xmc_scalelines_plotwin2event',$
                                    UVALUE='Plot Window 2')
    
       row_base = widget_base(state.xmc_scalelines_base,$
                              EVENT_PRO='xmc_scalelines_event',$
                              /ROW)
         
          xmin = coyote_field2(row_base,$
                               LABELFONT=buttonfont,$
                               FIELDFONT=textfont,$
                               TITLE='X Min:',$
                               UVALUE='X Min',$
                               XSIZE=12,$
                               EVENT_PRO='xmc_scalelines_minmaxevent',$
                               /CR_ONLY,$
                               TEXTID=textid)
          state.xmin_fld = [xmin,textid]
          
          xmax = coyote_field2(row_base,$
                               LABELFONT=buttonfont,$
                               FIELDFONT=textfont,$
                               TITLE='X Max:',$
                               UVALUE='X Max',$
                               XSIZE=12,$
                               EVENT_PRO='xmc_scalelines_minmaxevent',$
                               /CR_ONLY,$
                               TEXTID=textid)
          state.xmax_fld = [xmax,textid]
          
          ymin = coyote_field2(row_base,$
                               LABELFONT=buttonfont,$
                               FIELDFONT=textfont,$
                               TITLE='Y Min:',$
                               UVALUE='Y Min',$
                               XSIZE=12,$
                               EVENT_PRO='xmc_scalelines_minmaxevent',$
                               /CR_ONLY,$
                               TEXTID=textid)
          state.ymin_fld = [ymin,textid]
          
          ymax = coyote_field2(row_base,$
                               LABELFONT=buttonfont,$
                               FIELDFONT=textfont,$
                               TITLE='Y Max:',$
                               UVALUE='Y Max',$
                               XSIZE=12,$
                               EVENT_PRO='xmc_scalelines_minmaxevent',$
                               /CR_ONLY,$
                               TEXTID=textid)
          state.ymax_fld = [ymax,textid]
          
    button = widget_button(state.xmc_scalelines_base,$
                           FONT=buttonfont,$
                           EVENT_PRO='xmc_scalelines_event',$
                           VALUE='Accept',$
                           UVALUE='Accept')
    
; Get things running.  Center the widget using the Fanning routine.
      
    cgcentertlb,state.xmc_scalelines_base
    widget_control, state.xmc_scalelines_base, /REALIZE
    
;  Get plotwin ids
    
    widget_control, state.plotwin1, GET_VALUE=x
    state.plotwin1_wid = x

    widget_control, state.plotwin2, GET_VALUE=x
    state.plotwin2_wid = x
    
    window,/FREE,/PIXMAP,XSIZE=state.plot1size[0],YSIZE=state.plot1size[1]
    state.pixmap1_wid = !d.window

    window,/FREE,/PIXMAP,XSIZE=state.plot2size[0],YSIZE=state.plot2size[1]
    state.pixmap2_wid = !d.window

;  Get sizes for things.
    
    widget_geom = widget_info(state.xmc_scalelines_base, /GEOMETRY)
    state.buffer[0]=widget_geom.xsize-state.plot1size[0]
    state.buffer[1]=widget_geom.ysize-state.plot1size[1]-$
      state.plot2size[1]

;  Get things running

    xmc_scalelines_selectspec
    xmc_scalelines_makescale
    xmc_scalelines_maketelluric
    xmc_scalelines_makespec
    xmc_scalelines_getminmax
    xmc_scalelines_setminmax
    xmc_scalelines_plotupdate1
    xmc_scalelines_plotupdate2

; Start the Event Loop. This will be a non-blocking program.
    
    XManager, 'xmc_scalelines', $
              state.xmc_scalelines_base, $
              EVENT_HANDLER='xmc_scalelines_resizeevent',$
              CLEANUP='xmc_scalelines_cleanup'
    
    if n_elements(PARENT) ne 0 then widget_control, parent, SENSITIVE=1

;  Now return required info

    scales = fltarr(n_elements(state.std[*,0,0]),state.norders)
    for i = 0, state.norders-1 do begin

       if n_elements((*state.cpoints).(i)[0,*]) eq 2 then begin

          scales[*,i] = 1.0
          
       endif else begin
          
          scales[*,i] = spline((*state.cpoints).(i)[0,*],$
                               (*state.cpoints).(i)[1,*],$
                               state.std[*,0,i],state.tension)
          

       endelse
       
        
    endfor
    cancel = state.cancel
    vshift = state.vshift 
    cutreg = state.cutreg

    ptr_free, state.scalespec
    ptr_free, state.tellspec

    state = 0B

endif

end

