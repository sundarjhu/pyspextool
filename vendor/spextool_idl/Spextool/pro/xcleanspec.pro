pro xcleanspec_initcommon

  common xcleanspec_state, state

end
;
;==============================================================================
;
pro xcleanspec_chooseaperture

  common xcleanspec_state

  flxyranges = fltarr(2,state.norders)
  uncyranges = fltarr(2,state.norders)
  snryranges = fltarr(2,state.norders)
  
; Get plot ranges 
  
  for i = 0, state.norders-1 do begin

     idx = i*state.naps+state.ap
     
     wav = reform((*state.workspec)[*,0,idx])
     flx = reform((*state.workspec)[*,1,idx])
     unc = reform((*state.workspec)[*,2,idx])

;  Wavelengths
     
     (*state.xranges)[*,i] = [min(wav,MAX=max,/NAN),max]
     (*state.cenwave)[i] = mean(wav,/NAN)
     
;  Fluxes

     x = findgen(n_elements(flx))
     smooth = mc_robustsg(x,flx,5,3,0.1,CANCEL=cancel)
     if cancel then return
     
     min = min(smooth[*,1],/NAN,MAX=max)
     flxyranges[*,i] = mc_bufrange([min,max],0.05)

;  Uncertainties

     smooth = mc_robustsg(x,unc,5,3,0.1,CANCEL=cancel)
     if cancel then return

     min = min(smooth[*,1],/NAN,MAX=max)
     uncyranges[*,i] = mc_bufrange([min,max],0.05)

;  S/N

     smooth = mc_robustsg(x,flx/unc,5,3,0.1,CANCEL=cancel)
     if cancel then return
     
     min = min(smooth[*,1],/NAN,MAX=max)
     snryranges[*,i] = mc_bufrange([min,max],0.05)     

  endfor

;  Do it for the continuous case
  
  state.flxxrange = [min(*state.xranges,MAX=max),max]
  state.flxyrange = [min(flxyranges,MAX=max),max]
  state.absflxyrange = state.flxyrange
  *state.flxyranges = flxyranges
  
  state.uncxrange = state.flxxrange
  state.uncyrange = [min(uncyranges,MAX=max),max]
  state.absuncyrange = state.uncyrange
  *state.uncyranges = uncyranges
  
  state.snrxrange = state.flxxrange
  state.snryrange = [min(snryranges,MAX=max),max]
  state.abssnryrange = state.snryrange
  *state.snryranges = snryranges
  
  case state.spectype of

     'Flux': begin

        state.xrange = state.flxxrange
        state.yrange = state.flxyrange
        state.ytitle =state.ytitles[0]
        *state.yranges = *state.flxyranges
        
     end

     'Uncertainty': begin

        state.xrange = state.uncxrange
        state.yrange = state.uncyrange
        state.absyrange = state.yrange
        state.ytitle =state.ytitles[1]
        *state.yranges = *state.uncyranges
        
     end        

     'S/N': begin

        state.xrange = state.snrxrange
        state.yrange = state.snryrange
        state.absyrange = state.yrange
        state.ytitle = state.ytitles[2]
        *state.yranges = *state.snryranges
        
     end
     
  endcase

  state.absxrange = state.xrange
  state.absyrange = state.yrange
    
;  Reset a bunch of arrays

  state.reg = !values.f_nan

  widget_control,state.mbut_fix,SET_BUTTON=0
  widget_control,state.mbut_rmv,SET_BUTTON=0
  

  if state.norders eq 1 then begin

     widget_control, state.edit_base, SENSITIVE=1
     widget_control, state.selectord_dl,SENSITIVE=0
     state.cliporder = 0
     
  endif else begin

     widget_control, state.edit_base, SENSITIVE=0
     widget_control, state.selectord_dl,SENSITIVE=1
     state.cliporder = -1
     
  endelse

  state.cursormode = 'None'
  
end
;
;==============================================================================
;
pro xcleanspec_cleanup,base

  common xcleanspec_state
  
  if n_elements(state) ne 0 then begin

     ptr_free, state.origfixmask
     ptr_free, state.origrmvmask
     ptr_free, state.origspec  
     ptr_free, state.hdr     
     ptr_free, state.orders 
     ptr_free, state.xranges
     ptr_free, state.yranges
     ptr_free, state.workspec
     ptr_free, state.workfixmask
     ptr_free, state.workrmvmask
     ptr_free, state.cenwave
     ptr_free, state.awave
     ptr_free, state.atrans
     
     ptr_free, state.uncyranges
     ptr_free, state.flxyranges
     ptr_free, state.snryranges
     
  endif
  
  state = 0B

end
;
;=============================================================================
;
pro xcleanspec_loadspec

  common xcleanspec_state

;  Get files.

  file = mc_cfld(state.ispectrum_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  file = mc_cfile(file,WIDGET_ID=state.xcleanspec_base,CANCEL=cancel)
  if cancel then return

;  Copy root of file to output field

  widget_control, state.oname_fld[1], $
                  SET_VALUE='c'+strtrim(file_basename(file,'.fits'),2)
  mc_setfocus,state.oname_fld,/LEFT,CANCEL=cancel
  if cancel then return
  
;  Read spectra
  
  mc_readspec,file,spc,hdr,obsmode,start,stop,norders,naps,orders,xunits, $
              yunits,slith_pix,slith_arc,slitw_pix,slitw_arc,rp,airmass, $
              xtitle,ytitle,CANCEL=cancel
  if cancel then return
  spc = mc_caaspecflags(spc,CANCEL=cancel)
  if cancel then return
  
;  Save info and set up some arrays
  
  *state.origspec = spc
  *state.workspec = spc
  *state.hdr      = hdr          
  *state.orders   = orders
  state.norders   = norders
  state.naps      = naps
  state.ap        = 0
  *state.xranges  = fltarr(2,norders,/NOZERO)
  *state.cenwave  = fltarr(norders,/NOZERO)
  state.snrcut     = 0.0
  
; Create the masks
  
  npixels            = n_elements((*state.origspec)[*,0,0])
  *state.origfixmask = make_array(npixels,state.norders*state.naps,/BYTE,VAL=1)
  *state.origrmvmask = make_array(npixels,state.norders*state.naps,/BYTE,VAL=1)
  *state.workfixmask = make_array(npixels,state.norders*state.naps,/BYTE,VAL=1)
  *state.workrmvmask = make_array(npixels,state.norders*state.naps,/BYTE,VAL=1)

;  Deal with the units
  
  state.xtitle = xtitle
  lidx = strpos(ytitle,'(')
  ridx = strpos(ytitle,')')
  ypunits = strmid(ytitle,lidx+1,ridx-lidx-1)
  state.ytitles = [ytitle,'!5Uncertainty ('+ypunits+')','!5S/N']
  
;  Update droplists and fields
  
  widget_control,state.ap_dl,$
                 SET_VALUE=strtrim(string(indgen(naps)+1,FORMAT='(I2)'),2)
  widget_control, state.ap_dl, SENSITIVE=(state.naps eq 1) ? 0:1

  val = ['None',strtrim(string(orders,FORMAT='(I3)'),2)]
  widget_control,state.selectord_dl,SET_VALUE=val  

  widget_control, state.snr_fld[1],SET_VALUE=string(state.snrcut,FORMAT='(I1)')
  
;  Update the smoothing fields
  
  scale = (slitw_pix gt 2) ? 1.5:2.5
  
  widget_control, state.sgwin_fld[1],$
                  SET_VALUE=strtrim(string(slitw_pix*scale,FORMAT='(f4.1)'),2)
  
  widget_control, state.fwhm_fld[1],SET_VALUE=strtrim(slitw_pix,2)

;   Sensitize the smoothing box

  widget_control, state.box5_base,SENSITIVE=1
  
;  Load the atmospheric transmission

  if rp ne 0 then begin

;  Get the resolutions available

     files = file_basename(file_search(filepath('atran*.fits', $
                                                ROOT_DIR=state.spextoolpath, $
                                                SUBDIR='data')))
     nfiles = n_elements(files)
     rps = lonarr(nfiles)
     for i =0,nfiles-1 do rps[i] = long(strmid( $
        file_basename(files[i],'.fits'),5))

;  Find the one closest to our observations
     
     min = min(abs(rps-rp),idx)

;  Load the file and make sure the atmosphere button is sensitive
     
     spec = readfits(filepath('atran'+strtrim(rps[idx],2)+'.fits', $
                              ROOT_DIR=state.spextoolpath, $
                              SUBDIR='data'),/SILENT)
     
     *state.awave = reform(spec[*,0])
     *state.atrans = reform(spec[*,1])
     widget_control, state.mbut_atmos,/SENSITIVE
     
  endif else begin

;  Nope.  Close things off from access
     
     *state.awave = 0B
     *state.atrans = 0B
     state.plotatmosphere = 0
     widget_control, state.mbut_atmos,SET_BUTTON=0
     widget_control, state.mbut_atmos,SENSITIVE=0
     
  endelse

;  Get things going

  widget_control, state.mbut_flx,/SET_BUTTON
  xcleanspec_chooseaperture
  xcleanspec_setminmax
  xcleanspec_plotspec

;  Unfreeze the widget
  
  state.freeze = 0
  
end
;
;===============================================================================
;
pro xcleanspec_pickspec,current,new

  common xcleanspec_state
  
;  Store current plot range values into proper spectype

  case current of

     'Flux': begin

        state.flxxrange = state.xrange
        state.flxyrange = state.yrange
        state.absflxyrange = state.absyrange

     end

     'Uncertainty': begin

        state.uncxrange = state.xrange
        state.uncyrange = state.yrange
        state.absuncyrange = state.absyrange
        
     end

     'S/N': begin

        state.snrxrange = state.xrange
        state.snryrange = state.yrange
        state.abssnryrange = state.absyrange

     end

  endcase

;  Determine new spectrum and ranges
  
  case new of

     'Flux': begin

        state.ytitle = state.ytitles[0]
        *state.yranges = *state.flxyranges

        if total(state.flxxrange - state.xrange) eq 0 then begin
           
           state.xrange = state.flxxrange
           state.yrange = state.flxyrange
           
        endif else begin
           
;  Compute a new yrange based on the plot window wrange
           
           tmp = 0
           for i = 0,state.norders-1 do begin

              z = where((*state.workspec)[*,0,i] gt state.xrange[0] and $
                        (*state.workspec)[*,0,i] lt state.xrange[1],cnt)
              if cnt ne 0 then tmp = [tmp,(*state.workspec)[z,1,i]]
              
           endfor
           tmp = tmp[1:*]
           min = min(tmp,MAX=max,/NAN)
           state.flxyrange = mc_bufrange([min,max],0.05)
           state.yrange = state.flxyrange
           state.flxxrange = state.xrange

        endelse
           
;  Store absolute ranges

        state.absyrange = state.absflxyrange
           
     end

     'Uncertainty': begin

        state.ytitle = state.ytitles[1]
        *state.yranges = *state.uncyranges
        
        if total(state.uncxrange - state.xrange) eq 0 then begin

           state.xrange = state.uncxrange
           state.yrange = state.uncyrange
           
        endif else begin

;  Compute a new yrange based on the plot window wrange

           tmp = 0
           for i = 0,state.norders-1 do begin

              z = where((*state.workspec)[*,0,i] gt state.xrange[0] and $
                        (*state.workspec)[*,0,i] lt state.xrange[1],cnt)
              if cnt ne 0 then tmp = [tmp,(*state.workspec)[z,2,i]]

           endfor
           tmp = tmp[1:*]
           min = min(tmp,MAX=max,/NAN)
           state.uncyrange = mc_bufrange([min,max],0.05)
           state.yrange = state.uncyrange
           state.uncxrange = state.xrange
          
        endelse

;  Store absolute ranges

        state.absyrange = state.absuncyrange
           
     end

     'S/N': begin

        state.ytitle = state.ytitles[2]
        *state.yranges = *state.snryranges
        
        if total(state.snrxrange - state.xrange) eq 0 then begin

           state.xrange = state.snrxrange
           state.yrange = state.snryrange
           
        endif else begin

;  Compute a new yrange based on the plot window wrange

           tmp = 0
           for i = 0,state.norders-1 do begin
              
              z = where((*state.workspec)[*,0,i] gt state.xrange[0] and $
                        (*state.workspec)[*,0,i] lt state.xrange[1],cnt)
              if cnt ne 0 then begin

                 tmp = [tmp,(*state.workspec)[z,1,i]/(*state.workspec)[z,2,i]]

              endif
              
           endfor
           tmp = tmp[1:*]
           min = min(tmp,MAX=max,/NAN)
           state.snryrange = mc_bufrange([min,max],0.05)
           state.yrange = state.snryrange
           state.snrxrange = state.xrange
          
        endelse

;  Store absolute ranges

        state.absyrange = state.abssnryrange

     end

  endcase

  state.spectype = new  

end
;
;===============================================================================
;
pro xcleanspec_plotspec

  common xcleanspec_state

  wset, state.pixmap_wid
  erase,COLOR=20
  
; Get plot position.  This is the one for multi-order, no atmosphere
  
  position = [120,60,state.plotwin_size[0]-20,state.plotwin_size[1]-100]

;  Adjust if we are doing just one order  
  
  if state.norders eq 1 then position[3] = position[3]+40
  
;  Are we plotting the atmosphere?

  noerase = 0
  ystyle = 1
  if state.plotatmosphere then begin

     position[2] = position[2]-30
     plot,[1],[1],XSTYLE=5,/NODATA,YRANGE=[0,1],YSTYLE=5, $
          CHARSIZE=state.charsize,XRANGE=state.xrange,BACKGROUND=20, $
          POSITION=position,/DEVICE

     z = where(*state.awave ge state.xrange[0] and $
               *state.awave le state.xrange[1])
     oplot,(*state.awave)[z],(*state.atrans)[z],COLOR=5,PSYM=10
     ystyle = 9
     noerase = 1

  endif else noerase = 0

;  Plot the main plot
  
  plot,[1],[1],/XSTYLE,XTITLE=state.xtitle,YTITLE=state.ytitle, $
       CHARSIZE=state.charsize,/NODATA,YRANGE=state.yrange, $
       XRANGE=state.xrange,YSTYLE=ystyle,BACKGROUND=20,NOERASE=noerase, $
       POSITION=position,/DEVICE

;  Check to see whether we are plotting one or multiple orders

  if state.norders gt 1 then begin
  
     xyouts,15,state.plotwin_size[1]-20,'!5Order',ALIGNMENT=0,/DEVICE, $
            CHARSIZE=state.charsize
     xyouts,15,state.plotwin_size[1]-55,'!5Fix Mask',ALIGNMENT=0,/DEVICE, $
            CHARSIZE=state.charsize,COLOR=(state.cursormode eq 'Fix') ? 7:1
     xyouts,15,state.plotwin_size[1]-80,'!5Remove Mask',ALIGNMENT=0,/DEVICE, $
            CHARSIZE=state.charsize,COLOR=(state.cursormode eq 'Remove') ? 7:1
     offsets = [55,80]
     
  endif else begin

     xyouts,15,state.plotwin_size[1]-20,'!5Fix Mask',ALIGNMENT=0,/DEVICE, $
            CHARSIZE=state.charsize,COLOR=(state.cursormode eq 'Fix') ? 7:1
     xyouts,15,state.plotwin_size[1]-45,'!5Remove Mask',ALIGNMENT=0,/DEVICE, $
            CHARSIZE=state.charsize,COLOR=(state.cursormode eq 'Remove') ? 7:1
     offsets = [20,45]

  endelse

;  Now start the loop
  
  l = 0
  spectra = *state.workspec
  for i = 0, state.norders-1 do begin

     idx = i*state.naps+state.ap
     if ~mc_rangeinrange((*state.xranges)[*,i],!x.crange) then continue

     wave   = spectra[*,0,idx]
     z      = where(finite(wave) eq 1)
     wave   = wave[z]
     cflux  = spectra[z,1,idx]
     cerror = spectra[z,2,idx]
     flag   = spectra[z,3,idx]
     
     case state.spectype of 
        
        'Flux': spec = cflux
        
        'Uncertainty': spec = cerror
        
        'S/N': spec = cflux/cerror
        
     endcase
     
;  Plot spectra

        case state.altcolor of
           
           2: color = (i mod 2) ? 1:3
           
           3: begin
              
              case i mod 3 of
                 
                 0: color=3
                 
                 1: color=1
                 
                 2: color=2
                 
              endcase
              
           end
           
        endcase   
     
     if state.cliporder ne -1 then begin
        
        if i eq state.cliporder then begin
           
           savewave = wave
           saveflux = spec
           savecolor = color
           
        endif
        oplot,wave,spec,COLOR=75,PSYM=10
        
     endif else oplot,wave,spec,COLOR=color,PSYM=10
     
;  label orders

     if state.norders gt 1 then begin
     
        min = max([!x.crange[0],(*state.xranges)[0,i]])
        max = min([!x.crange[1],(*state.xranges)[1,i]])
        
        lwave = (min+max)/2.
        
        if lwave gt !x.crange[0] then begin
           
           xy = convert_coord(lwave,!y.crange[1],/DATA,/TO_DEVICE)
           
           xyouts,xy[0],state.plotwin_size[1]-15-15*(i mod 2), $
                  strtrim(string((*state.orders)[i],FORMAT='(I3)'),2), $
                  /DEVICE,COLOR=color,ALIGNMENT=0.5,CHARSIZE=state.charsize
           
        endif
           
     endif
     
;  Plot masks
     
     zsnr = where((*state.workfixmask)[*,idx] eq 0,cntsnr)
     zusr = where((*state.workrmvmask)[*,idx] eq 0,cntusr)
     
     if cntsnr ne 0 then begin
        
        xy = convert_coord(spectra[zsnr,0,idx],!y.crange[1],/DATA,/TO_DEVICE)
        
        plots,reform(xy[0,*]),state.plotwin_size[1]-offsets[0]+5,PSYM=1, $
              COLOR=color,/DEVICE
        
     endif
     
     if cntusr ne 0 then begin
        
        xy = convert_coord(spectra[zusr,0,idx],!y.crange[1],/DATA,/TO_DEVICE)
        plots,reform(xy[0,*]),state.plotwin_size[1]-offsets[1]+5,PSYM=1, $
              COLOR=color,/DEVICE
        
     endif

;  Plot flags if requested

     if state.plotlincor then begin
        
        mask = mc_bitset(fix(flag),0,CANCEL=cancel)
        z = where(mask eq 1,cnt)
        plotsym,0,0.8,/FILL
        if cnt ne 0 then oplot,wave[z],spec[z],PSYM=8,COLOR=2
        
     endif
     
     if state.plotfixed then begin
        
        mask = mc_bitset(fix(flag),2,CANCEL=cancel)
        z = where(mask eq 1,cnt)
        plotsym,0,1.0,/FILL
        if cnt ne 0 then oplot,wave[z],spec[z],PSYM=8,COLOR=7
        
     endif
     l = l + 1
     
  endfor
  
  if state.cliporder ne -1 then begin
     
     oplot,savewave,saveflux,COLOR=savecolor,PSYM=10

;  Plot flags if requested

     if state.plotlincor then begin
        
        mask = mc_bitset(fix(flag),0,CANCEL=cancel)
        z = where(mask eq 1,cnt)
        plotsym,0,0.8,/FILL
        if cnt ne 0 then oplot,wave[z],spec[z],PSYM=8,COLOR=2
        
     endif
     
     if state.plotfixed then begin
        
        mask = mc_bitset(fix(flag),2,CANCEL=cancel)
        z = where(mask eq 1,cnt)
        plotsym,0,1.0,/FILL
        if cnt ne 0 then oplot,wave[z],spec[z],PSYM=8,COLOR=7
        
     endif
              
  endif

  if ystyle eq 9 then begin

     ticks = ['0.0','0.2','0.4','0.6','0.8','1.0']
     axis,YAXIS=1,YTICKS=5,YTICKNAME=ticks,YMINOR=2,COLOR=5, $
          CHARSIZE=state.charsize

  endif
  
;  Draw zero line
  
  if !y.crange[0] lt 0 then plots,!x.crange,[0,0],LINESTYLE=1

;  Store scaling
  
  state.xscale = !x
  state.yscale = !y
  state.pscale = !p

;  Copy the pixmap to the display
  
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plotwin_size[0],state.plotwin_size[1],0,0, $
                state.pixmap_wid]
  
end
;
;=============================================================================
;
pro xcleanspec_setminmax

  common xcleanspec_state
  
  widget_control, state.xmin_fld[1], SET_VALUE=strtrim(state.xrange[0],2)
  widget_control, state.xmax_fld[1], SET_VALUE=strtrim(state.xrange[1],2)

  widget_control, state.ymin_fld[1], SET_VALUE=strtrim(state.yrange[0],2)
  widget_control, state.ymax_fld[1], SET_VALUE=strtrim(state.yrange[1],2)

  xcleanspec_setslider

end
;
;===============================================================================
;
pro xcleanspec_setslider

  common xcleanspec_state

;  Get new slider value
  
  del = state.absxrange[1]-state.absxrange[0]
  midwave = (state.xrange[1]+state.xrange[0])/2.
  state.sliderval = round((midwave-state.absxrange[0])/del*100)
     
  widget_control, state.slider, SET_VALUE=state.sliderval
  
end
;
;===============================================================================
;
pro xcleanspec_smoothspec

  common xcleanspec_state

  case state.smoothtype of

     'Savitzky-Golay': begin
        
        sgwin = mc_cfld(state.sgwin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        
        sgdeg = mc_cfld(state.sgdeg_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        
        sgdeg = mc_crange(sgdeg,sgwin,'SG Degree',/KLT,$
                          WIDGET_ID=state.xcleanspec_base,CANCEL=cancel)
        if cancel then return
        
        print, ' '
        print, 'Performing a Savitzky-Golay smoothing with a window of '+$
               strtrim(sgwin,2)
        print, ' and a degree of '+strtrim(fix(sgdeg),2)+'.'
        print, ' '
        
        for i = 0, state.norders-1 do begin
           
           for j = 0, state.naps-1 do begin
              
              idx = i*state.naps+j
              
              z = where(finite((*state.workspec)[*,1,idx]) eq 1)
              
              cflux  = (*state.workspec)[z,1,idx]
              cerror = (*state.workspec)[z,2,idx]
              
              cflux = savitzky_golay(cflux,sgwin,DEGREE=sgdeg,ERROR=cerror)
              
              (*state.workspec)[z,1,idx] = cflux
              (*state.workspec)[z,2,idx] = cerror
              
           endfor
           
        endfor
        
        if state.norders gt 1 or state.naps gt 1 then begin
           
           state.smthhistory='These spectra have been convolved with ' + $
                             'a Savitzky-Golay filter of width '+ $
                             strtrim(sgwin,2)+' pixels and degree '+ $
                             strtrim(sgdeg,2)+'.'
           
        endif else begin
           
           state.smthhistory='This spectrum has been convolved with ' + $
                             'a Savitzky-Golay filter of width '+ $
                             strtrim(sgwin,2)+' pixels and degree '+ $
                             strtrim(sgdeg,2)+'.'
           
        endelse

     end

     'Gaussian': begin
        
        fwhm = mc_cfld(state.fwhm_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        
        print, ' '
        print, 'Smoothing with a Gaussian with FWHM='+ $
               string(fwhm,format='(f3.1)')+'.'
        print, ' '
        
        for i = 0, state.norders-1 do begin
           
           for j = 0, state.naps-1 do begin
              
              idx = i*state.naps+j
              
              z = where(finite((*state.workspec)[*,1,idx]) eq 1,cnt)
              mc_convolvespec,findgen(cnt),(*state.workspec)[z,1,idx],fwhm, $
                              cflux,cerror,ERROR=(*state.workspec)[z,2,idx], $
                              CANCEL=cancel
              if cancel then return
              
              (*state.workspec)[z,1,idx] = cflux
              (*state.workspec)[z,2,idx] = cerror
              
           endfor
           
        endfor
        
        if state.norders gt 1 or state.naps gt 1 then begin
           
           state.smthhistory = 'These spectra have been convolved with a ' + $
                               'Gaussian of FWHM='+strtrim(fwhm,2)+' pixels.'
           
        endif else begin
           
           state.smthhistory = 'This spectrum has been convolved with a ' + $
                               'Gaussian of FWHM='+strtrim(fwhm,2)+' pixels.'
           
        endelse

     end
        
  endcase

  *state.origspec = *state.workspec

  widget_control, state.box5_base,SENSITIVE=0
  
end
;
;===============================================================================
;
pro xcleanspec_sncut

  common xcleanspec_state

  snrcut = mc_cfld(state.snr_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return

  snrcut = mc_crange(snrcut,0,'S/N Cut',/KGE, $
                     WIDGET_ID=state.xcleanspec_base,CANCEL=cancel)
  if cancel then return

  state.snrcut = snrcut
  for i = 0,state.norders-1 do begin

     for j = 0,state.naps-1 do begin

        idx = i*state.naps+j

        y = (*state.origspec)[*,1,idx]
        u = (*state.origspec)[*,2,idx]
        m = (*state.workrmvmask)[*,idx]
        
        zbad = where(abs(y/u) lt snrcut,cnt)
                  
        if cnt ne 0 then begin

           y[zbad] = !values.f_nan
           u[zbad] = !values.f_nan
           m[zbad] = 0
           
        endif else m[*] = 1
        
        (*state.workspec)[*,1,idx] = y
        (*state.workspec)[*,2,idx] = u
        (*state.workrmvmask)[*,idx] = m
        
     endfor
     
  endfor

;  Store the results
  
  *state.origrmvmask = *state.workrmvmask 
  *state.origfixmask = *state.workfixmask
  
end
;
;
;===============================================================================
;
pro xcleanspec_writefile

  common xcleanspec_state
  
  ifile = mc_cfld(state.ispectrum_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return

  file = mc_cfld(state.oname_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
 
  hdr = *state.hdr
  fxaddpar,hdr,'SNRCUT',state.snrcut,' Xcleanspec S/N cut',BEFORE='HISTORY'

  if state.smthhistory ne '' then begin

     sxaddhist,' ',hdr
     sxaddhist,'####################### Xcleanspec History ' + $
               '########################',hdr
     sxaddhist,' ',hdr
     
     history = mc_splittext(state.smthhistory,67,CANCEL=cancel)
     if cancel then return
     sxaddhist,history,hdr
     
  endif
  
  writefits, state.path+file+'.fits',*state.workspec,hdr
  xvspec,state.path+file+'.fits',/PLOTLINMAX,/PLOTFIX

  end
;
;===============================================================================
;
pro xcleanspec_zoom,IN=in,OUT=out

  common xcleanspec_state

  delabsx = state.absxrange[1]-state.absxrange[0]
  delx    = state.xrange[1]-state.xrange[0]
  
  delabsy = state.absyrange[1]-state.absyrange[0]
  dely    = state.yrange[1]-state.yrange[0]
  
  xcen = state.xrange[0]+delx/2.
  ycen = state.yrange[0]+dely/2.
  
    case state.cursormode of 
        
        'XZoom': begin
            
            z = alog10(delabsx/delx)/alog10(2)
            if keyword_set(IN) then z = z+1 else z=z-1
            hwin = delabsx/2.^z/2.
            state.xrange = [xcen-hwin,xcen+hwin]
            
        end
        
        'YZoom': begin
            
            z = alog10(delabsy/dely)/alog10(2)
            if keyword_set(IN) then z = z+1 else z=z-1
            hwin = delabsy/2.^z/2.
            state.yrange = [ycen-hwin,ycen+hwin]
            
        end
        
        'Zoom': begin
            
            z = alog10(delabsx/delx)/alog10(2)
            if keyword_set(IN) then z = z+1 else z=z-1
            hwin = delabsx/2.^z/2.
            state.xrange = [xcen-hwin,xcen+hwin]
            
            z = alog10(delabsy/dely)/alog10(2)
            if keyword_set(IN) then z = z+1 else z=z-1
            hwin = delabsy/2.^z/2.
            state.yrange = [ycen-hwin,ycen+hwin]
            
        end
        
        else:
        
    endcase

    xcleanspec_plotspec
    xcleanspec_setminmax

    
end
;
;=============================================================================
;
;----------------------------- Event Handlder -------------------------------
;
;=============================================================================
;
pro xcleanspec_event, event

  common xcleanspec_state
  
;  Check to see if it is the help file 'Done' Button
  
  widget_control, event.id,  GET_UVALUE = uvalue
  
  if uvalue eq 'Quit' then begin
     
     widget_control, event.top, /DESTROY
     return
     
  endif

;  Must be a main widget event
  
  case uvalue of

     '2 Color Button': begin

        state.altcolor = 2
        if ~state.freeze then xcleanspec_plotspec
        
     end

     '3 Color Button': begin

        state.altcolor = 3
        if ~state.freeze then xcleanspec_plotspec
        
     end

     'Aperture': begin

        
        state.ap = event.index
        xcleanspec_chooseaperture
        xcleanspec_setminmax
        xcleanspec_plotspec        

     end
     
     'Fix Button': begin

        if state.freeze then return        
        if state.cliporder eq -1 then return
        if state.spectype ne 'Flux' then return        
        state.cursormode = 'Fix'
        state.reg = !values.f_nan                
        widget_control, state.mbut_fix,SET_BUTTON=1
        xcleanspec_plotspec
        
     end
          
    'Help':  begin

       pre = (strupcase(!version.os_family) eq 'WINDOWS') ? 'start ':'open '
       
       spawn, pre+filepath(strlowcase(state.instrument)+'_spextoolmanual.pdf', $
                           ROOT=state.packagepath,$
                            SUBDIR='manual')
       
    end
     
     'Input Spectrum': begin
        
        obj = dialog_pickfile(DIALOG_PARENT=state.xcleanspec_base,$
                              PATH=state.path, GET_PATH=path,$
                              /MUST_EXIST,FILTER='*.fits')
        if obj ne '' then begin
           
           widget_control,state.ispectrum_fld[1],SET_VALUE = strtrim(obj,2)
           mc_setfocus,state.ispectrum_fld
           state.path = path

        endif
           
     end

     'Load Spectra': xcleanspec_loadspec
        
     'Plot Atmosphere Button': begin

        state.plotatmosphere = event.select
        if state.freeze then return
        xcleanspec_plotspec

     end

     'Plot Fixed Button': begin

        state.plotfixed = event.select
        if state.freeze then return
        xcleanspec_plotspec        
        
     end
     
     'Plot Flux Button': begin

        if state.freeze then return
        xcleanspec_pickspec,state.spectype,'Flux'        
        xcleanspec_setminmax
        xcleanspec_plotspec

     end

     'Plot Lincor Button': begin

        state.plotlincor = event.select
        if state.freeze then return
        xcleanspec_plotspec        

     end

     'Plot S/N Button': begin

        if state.freeze then return
        xcleanspec_pickspec,state.spectype,'S/N'        
        xcleanspec_setminmax
        xcleanspec_plotspec

     end

     'Plot Uncertainty Button': begin

        if state.freeze then return
        xcleanspec_pickspec,state.spectype,'Uncertainty'        
        xcleanspec_setminmax
        xcleanspec_plotspec
        
     end

     'Remove Button': begin

        state.cursormode = 'Remove'
        state.reg = !values.f_nan                
        widget_control, state.mbut_rmv,SET_BUTTON=1
        xcleanspec_plotspec
        
     end

     'Save Button': begin

        state.cursormode = 'None'
        state.reg = !values.f_nan
        *state.origspec = *state.workspec
        *state.origrmvmask = *state.workrmvmask 
        *state.origfixmask = *state.workfixmask
        widget_control, state.mbut_rmv,SET_BUTTON=0
        widget_control, state.mbut_fix,SET_BUTTON=0
        xcleanspec_plotspec
        
     end

     'Select Order Button': begin

        if state.freeze then return                
        state.cliporder = (event.index eq 0) ? -1:(event.index-1)
        if state.cliporder eq -1 then begin

           state.cursormode = 'None'
           widget_control, state.edit_base,SENSITIVE=0

        endif else widget_control, state.edit_base,SENSITIVE=1

        if state.norders gt 1 then begin
           
           state.xrange = mc_bufrange((*state.xranges)[*,state.cliporder],0.05)
           state.yrange = mc_bufrange((*state.yranges)[*,state.cliporder],0.05)
           xcleanspec_setminmax
           
        endif
        xcleanspec_plotspec
                  
     end
     
     'Slider': begin

        if state.freeze then return
        del = state.absxrange[1]-state.absxrange[0]
        oldcen = (state.xrange[1]+state.xrange[0])/2.
        newcen = state.absxrange[0]+del*(event.value/100.)
        
        
        state.xrange = state.xrange + (newcen-oldcen)
        xcleanspec_plotspec
        
        if event.drag eq 0 then xcleanspec_setminmax
        
     end

     'Smooth Spectra': begin

        if state.freeze then return
        xcleanspec_smoothspec
        xcleanspec_plotspec

     end

     'Smooth Type': begin

        if event.value eq 'Savitzky-Golay' then begin

           state.smoothtype = event.value
           widget_control, state.gs_base, MAP=0
           widget_control, state.sg_base, MAP=1


        endif

        if event.value eq 'Gaussian' then begin

           state.smoothtype = event.value
           widget_control, state.sg_base, MAP=0
           widget_control, state.gs_base, MAP=1
           
        endif

     end
     
     'S/N Cut Field': begin

        if state.freeze then return
        xcleanspec_sncut
        xcleanspec_plotspec

     end

     'Undo Button': begin
        
        *state.workspec = *state.origspec
        *state.workrmvmask = *state.origrmvmask
        *state.workfixmask = *state.origfixmask
        state.reg = !values.f_nan                
        xcleanspec_plotspec
        
     end
         
     'Write File': begin
        
        if state.freeze then return
        xcleanspec_writefile
        
     end

     else:
     
  endcase

getout:
  
end
;
;===============================================================================
;
pro xcleanspec_minmaxevent,event

  common xcleanspec_state

  widget_control, event.id,  GET_UVALUE = uvalue
  
  if state.freeze then goto, cont

  case uvalue of 
     
     'X Min': begin
        
        xmin = mc_cfld(state.xmin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then goto, cont
        xmin = mc_crange(xmin,state.xrange[1],'X Min',/KLT,$
                         WIDGET_ID=state.xcleanspec_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control, state.xmin_fld[0],SET_VALUE=state.xrange[0]
           goto, cont
           
        endif else state.xrange[0] = xmin
        
     end
     
     'X Max': begin
        
        xmax = mc_cfld(state.xmax_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then goto, cont
        xmax = mc_crange(xmax,state.xrange[0],'X Max',/KGT,$
                       WIDGET_ID=state.xcleanspec_base,CANCEL=cancel)
        if cancel then begin
            
            widget_control, state.xmax_fld[0],SET_VALUE=state.xrange[1]
            goto, cont
            
        endif else state.xrange[1] = xmax

    end

    'Y Min': begin

        ymin = mc_cfld(state.ymin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then goto, cont
        ymin = mc_crange(ymin,state.yrange[1],'Y Min',/KLT,$
                         WIDGET_ID=state.xcleanspec_base,CANCEL=cancel)
        if cancel then begin
           
            widget_control,state.ymin_fld[0],SET_VALUE=state.yrange[0]
            goto, cont
            
        endif else state.yrange[0] = ymin
        
    end
     
    'Y Max': begin
       
       ymax = mc_cfld(state.ymax_fld,4,/EMPTY,CANCEL=cancel)
       if cancel then return
       ymax = mc_crange(ymax,state.yrange[0],'Y Max',/KGT,$
                        WIDGET_ID=state.xcleanspec_base,CANCEL=cancel)
       if cancel then begin
          
          widget_control,state.ymax_fld[0],SET_VALUE=state.yrange[1]
          goto, cont
          
        endif else state.yrange[1] = ymax
        
    end

    
 endcase

;  Set slider and update plot

  xcleanspec_setslider
  xcleanspec_plotspec

cont:

end
;
;===============================================================================
;
pro xcleanspec_plotwinevent,event

  common xcleanspec_state

  widget_control, event.id,  GET_UVALUE=uvalue
  
  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin
     
     widget_control, state.plotwin,INPUT_FOCUS=event.enter

     wset, state.plotwin_wid
     device, COPY=[0,0,state.plotwin_size[0],state.plotwin_size[1],0,0,$
                   state.pixmap_wid]
     return

  endif

;  Check for arrow keys
  
  if event.type eq 6 and event.release eq 1 then begin

     if state.freeze then return
     
     case event.key of

        5: begin  ;  left

           if state.freeze then return

           del = (state.xrange[1]-state.xrange[0])*0.3
           oldcen = (state.xrange[1]+state.xrange[0])/2.
           newcen = oldcen-del
           
           if newcen lt state.absxrange[0] then return
           state.xrange = state.xrange + (newcen-oldcen)
           xcleanspec_setminmax
           xcleanspec_plotspec
           
        end

        6: begin  ;  right

           if state.freeze then return           

           del = (state.xrange[1]-state.xrange[0])*0.3
           oldcen = (state.xrange[1]+state.xrange[0])/2.
           newcen = oldcen+del
           
           if newcen gt state.absxrange[1] then return
           state.xrange = state.xrange + (newcen-oldcen)
           xcleanspec_setminmax
           xcleanspec_plotspec

        end
           
        else:
           
     endcase
        
  endif

;  Now check for ASCII characters
  
  if event.type eq 5 and event.release eq 1 then begin

     if state.freeze then return
     
     case strtrim(event.ch,2) of 

        'a': begin
           
           state.absxrange = state.xrange
           state.absyrange=state.yrange
           
        end
        
        'c': begin          
           
           state.cursormode = 'None'
           widget_control, state.mbut_fix,SET_BUTTON=0
           widget_control, state.mbut_rmv,SET_BUTTON=0
           state.reg = !values.f_nan                
           *state.workspec = *state.origspec
           *state.workrmvmask = *state.origrmvmask
           *state.workfixmask = *state.origfixmask
           xcleanspec_plotspec
           
        end

        'd': begin

           if state.norders eq 1 then return
           state.cliporder = -1
           widget_control, state.selectord_dl,SET_DROPLIST_SELECT=0
           widget_control, state.mbut_rmv,SET_BUTTON=0
           widget_control, state.mbut_fix,SET_BUTTON=0
           state.cursormode = 'None'
           widget_control, state.edit_base,SENSITIVE=0
           xcleanspec_plotspec

        end

        'f': begin

           if state.cliporder eq -1 then return
           if state.spectype ne 'Flux' then return
           state.cursormode = 'Fix'
           state.reg = !values.f_nan                
           widget_control, state.mbut_fix,SET_BUTTON=1
           xcleanspec_plotspec
           
        end
        
        'i': xcleanspec_zoom,/IN

        'm': begin

        end
        
        'o': xcleanspec_zoom,/OUT

        'q': begin

           widget_control, event.top, /DESTROY
           return

        end
        
        'r': begin

           if state.cliporder eq -1 then return
           state.cursormode = 'Remove'
           state.reg = !values.f_nan                
           widget_control, state.mbut_rmv,SET_BUTTON=1
           xcleanspec_plotspec
                      
        end
        
        's': begin

           if state.cliporder eq -1 then begin
           
;  Find which order you are talking about              
              
              xydev = convert_coord(*state.cenwave, $
                                    replicate(1.0,state.norders),$
                                    /DATA,/TO_DEVICE)
              min = min(abs(reform(xydev[0,*])-event.x),z)
              state.cliporder = z              
              widget_control, state.selectord_dl,SET_DROPLIST_SELECT=z+1

              if state.norders gt 1 then begin
                 
                 state.xrange = mc_bufrange((*state.xranges)[*,z],0.05)
                 state.yrange = mc_bufrange((*state.yranges)[*,z],0.05)
                 xcleanspec_setminmax
                 
              endif
              widget_control, state.edit_base,SENSITIVE=1
              xcleanspec_plotspec                
              
           endif else begin

              state.cursormode = 'None'
              state.reg = !values.f_nan
              *state.origspec = *state.workspec
              *state.origrmvmask = *state.workrmvmask 
              *state.origfixmask = *state.workfixmask
              widget_control, state.mbut_rmv,SET_BUTTON=0
              widget_control, state.mbut_fix,SET_BUTTON=0
              xcleanspec_plotspec
              
           endelse
              
        end

        'u': begin

           *state.workspec = *state.origspec
           *state.workrmvmask = *state.origrmvmask
           *state.workfixmask = *state.origfixmask
           state.reg = !values.f_nan                
           xcleanspec_plotspec

        end
        
        'w': begin
           
           state.xrange = state.absxrange
           state.yrange = state.absyrange
           xcleanspec_setminmax
           xcleanspec_plotspec
           
        end
        
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

  endif

  wset, state.plotwin_wid
     
  !p = state.pscale
  !x = state.xscale
  !y = state.yscale
  x  = event.x/float(state.plotwin_size[0])
  y  = event.y/float(state.plotwin_size[1])
  xy = convert_coord(x,y,/NORMAL,/TO_DATA,/DOUBLE)
  
  if event.type eq 1 then begin
     
     if state.freeze then return
     
     z = where(finite(state.reg) eq 1,count)
     if count eq 0 then begin
        
        wset, state.pixmap_wid
        state.reg[*,0] = xy[0:1]
        case state.cursormode of

           'Fix': begin

              plots,[xy[0],xy[0]],!y.crange,COLOR=7,THICK=2,LINESTYLE=2
              x = (*state.workspec)[*,0,state.cliporder*state.naps+state.ap]
              y = (*state.workspec)[*,1,state.cliporder*state.naps+state.ap]
              z = where(finite(x) eq 1)
              x = x[z]
              y = y[z]
              tabinv,x,xy[0],z
              z = mc_roundgt(z,CANCEL=cancel)
              if cancel then return
              plotsym,0,1,/FILL
              plots,x[z],y[z],PSYM=8,COLOR=7

           end           
           
           'XZoom': plots, [event.x,event.x],$
                           [0,state.plotwin_size[1]],COLOR=2,/DEVICE, $
                           LINESTYLE=2
           
           'YZoom': plots, [0,state.plotwin_size[0]],$
                           [event.y,event.y],COLOR=2,/DEVICE,LINESTYLE=2

           'Remove': begin

              plots,[xy[0],xy[0]],!y.crange,COLOR=7,THICK=2,LINESTYLE=2
              
           end
           
           else:
           
        endcase
        wset, state.plotwin_wid
        device, COPY=[0,0,state.plotwin_size[0], $
                      state.plotwin_size[1],0,0,state.pixmap_wid]
        
     endif else begin 
        
        state.reg[*,1] = xy[0:1]
        case state.cursormode of 

           'Fix': begin

              idx = state.cliporder*state.naps+state.ap
              
              x = (*state.workspec)[*,0,state.cliporder*state.naps+state.ap]
              y = (*state.workspec)[*,1,state.cliporder*state.naps+state.ap]
              z = where(finite(x) eq 1)
              x = x[z]
              y = y[z]

              waves = reform(state.reg[0,*])
              waves = waves[sort(waves)]
              
              tabinv,x,waves[0],zmin
              zmin = mc_roundgt(zmin,CANCEL=cancel)
              if cancel then return
              tabinv,x,waves[1],zmax
              zmax = mc_roundgt(zmax,CANCEL=cancel)
              if cancel then return

              x = [(*state.workspec)[zmin,0,idx],(*state.workspec)[zmax,0,idx]]
              y = [(*state.workspec)[zmin,1,idx],(*state.workspec)[zmax,1,idx]]
              u = [(*state.workspec)[zmin,2,idx],(*state.workspec)[zmax,2,idx]]
              
              cy = mc_polyfit1d(x,y,1,/SILENT)
              cu = mc_polyfit1d(x,u,1,/SILENT)
              
              (*state.workspec)[zmin:zmax,1,idx] = $
                 poly((*state.workspec)[zmin:zmax,0,idx],cy)
              
              (*state.workspec)[zmin:zmax,2,idx] = $
                 poly((*state.workspec)[zmin:zmax,0,idx],cu)
              
              (*state.workspec)[zmin:zmax,3,idx] = $
                 (*state.workspec)[zmin:zmax,3,idx]+4
              
              (*state.workfixmask)[zmin:zmax,idx] = 0
              
              state.reg = !values.f_nan
              
           end
              
           'Remove':  begin

              idx = state.cliporder*state.naps+state.ap
              
              min = min(state.reg[0,*],MAX=max)
              z = where((*state.workspec)[*,0,idx] gt min and $
                        (*state.workspec)[*,0,idx] lt max,cnt)
              if cnt ne 0 then begin

                 (*state.workspec)[z,1:2,idx] = !values.f_nan
                 (*state.workrmvmask)[z,idx] = 0

              endif
              state.reg = !values.f_nan
              xcleanspec_plotspec
              
           end

           'XZoom': begin

              state.xrange = [min(state.reg[0,*],MAX=max),max]
              state.cursormode = 'None'
              xcleanspec_setminmax
              
           end
           
           'YZoom': begin

              state.yrange = [min(state.reg[1,*],MAX=max),max]
              state.cursormode = 'None'
              xcleanspec_setminmax
              
           end
                                   
           'Zoom': begin
              
              state.xrange = [min(state.reg[0,*],MAX=max),max]
              state.yrange = [min(state.reg[1,*],MAX=max),max]
              state.cursormode = 'None'
              xcleanspec_setminmax

           end
           
           else:
           
        endcase
        xcleanspec_plotspec
        
     endelse
     
  endif
  
;  Copy the pixmaps and draw the cross hair or zoom lines.
  
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plotwin_size[0],state.plotwin_size[1],0,0,$
                state.pixmap_wid]
  
  case state.cursormode of 
     
     'XZoom': plots, [event.x,event.x],[0,state.plotwin_size[1]], $
                     COLOR=2,/DEVICE
     
     'YZoom': plots, [0,state.plotwin_size[0]],[event.y,event.y], $
                     COLOR=2,/DEVICE
     
     'Zoom': begin
        
        plots, [event.x,event.x],[0,state.plotwin_size[1]],COLOR=2,/DEVICE
        plots, [0,state.plotwin_size[0]],[event.y,event.y],COLOR=2,/DEVICE
        xy = convert_coord(event.x,event.y,/DEVICE,/TO_DATA,/DOUBLE)
        plots,[state.reg[0,0],state.reg[0,0]],[state.reg[1,0],xy[1]],$
              LINESTYLE=2,COLOR=2
        plots, [state.reg[0,0],xy[0]],[state.reg[1,0],state.reg[1,0]],$
               LINESTYLE=2,COLOR=2
        
     end

     'Remove': plots, [event.x,event.x],[0,state.plotwin_size[1]],COLOR=7, $
                      /DEVICE

     'Fix': plots, [event.x,event.x],[0,state.plotwin_size[1]],COLOR=7,/DEVICE

     'Undo': plots, [event.x,event.x],[0,state.plotwin_size[1]],COLOR=7,/DEVICE
     
     else: begin
        
        plots, [event.x,event.x],[0,state.plotwin_size[1]],COLOR=2,/DEVICE
        plots, [0,state.plotwin_size[0]],[event.y,event.y],COLOR=2,/DEVICE
        
     end
     
  endcase
  
end
;
;===============================================================================
;
pro xcleanspec_resizeevent,event

  common xcleanspec_state

  if n_params() eq 0 then begin
     
     size = widget_info(state.xcleanspec_base, /GEOMETRY)
     xsize = size.xsize
     ysize = size.ysize
     
  endif else begin
     
     widget_control, state.xcleanspec_base, TLB_GET_SIZE=size
     xsize = size[0]
     ysize = size[1]
     
  endelse

  state.plotwin_size[0] = xsize-state.winbuffer[0]
  state.plotwin_size[1] = ysize-state.winbuffer[1]
  
  widget_control, state.xcleanspec_base,UPDATE=0
  widget_control, state.plotwin, DRAW_XSIZE=state.plotwin_size[0], $
                  DRAW_YSIZE=state.plotwin_size[1]
  widget_control, state.xcleanspec_base,UPDATE=1
  
  

  wdelete,state.pixmap_wid
  window, /FREE, /PIXMAP,XSIZE=state.plotwin_size[0],YSIZE=state.plotwin_size[1]
  state.pixmap_wid = !d.window

  if state.freeze then begin

     erase, COLOR=20
     wset, state.plotwin_wid
     device, COPY=[0,0,state.plotwin_size[0],state.plotwin_size[1],0,0,$
                   state.pixmap_wid]
     
  endif else xcleanspec_plotspec

end  
;
;=============================================================================
;
;----------------------------- Main Program ---------------------------------
;
;=============================================================================
;
pro xcleanspec,instrument,CANCEL=cancel

  if ~xregistered('xcleanspec') then xcleanspec_initcommon

  common xcleanspec_state

  !except=0
  cleanplot,/SILENT
  void = check_math()
  !except = 1
  
;  Get spextool and instrument information 
  
  mc_getspextoolinfo,spextoolpath,packagepath,spextoolkeywords, $
                     instr,notspex,version,CANCEL=cancel
  if cancel then return
  
;  Load color table
  
  mc_mkct
  device, RETAIN=2
  
;  Get fonts
  
  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return

;  Get screen size

  screensize = get_screen_size()
     
;  Build the state structure

  state = {absxrange:[0.,0.],$
           absyrange:[0.,0.],$
           absflxyrange:[0.,0.],$
           abssnryrange:[0.,0.],$
           absuncyrange:[0.,0.],$
           altcolor:2,$
           ap:0,$
           ap_dl:0L,$
           atrans:ptr_new(2),$
           awave:ptr_new(2),$
           box5_base:0L,$
           buttonfont:buttonfont,$
           cenwave:ptr_new(2),$
           cliporder:-1,$
           charsize:1.5,$
           cursormode:'None',$
           flxxrange:[0.,0.],$
           flxyrange:[0.,0.],$
           flxyranges:ptr_new(2),$
           freeze:1,$
           fwhm_fld:[0L,0L],$
           gs_base:0L,$
           instrument:instr.instrument,$
           ispectrum_fld:[0L,0L],$
           hdr:ptr_new(2),$
           mbut_lincor:0L,$
           mbut_atmos:0L,$
           mbut_fixed:0L,$
           mbut_flx:0L,$
           mbut_snr:0L,$
           mbut_unc:0L,$
           mbut_fix:0L,$
           mbut_rmv:0L,$
           mbut_save:0L,$
           mbut_undo:0L,$
           message:0L,$
           oname_fld:[0L,0L],$
           orders:ptr_new(2),$
           origfixmask:ptr_new(2),$
           origrmvmask:ptr_new(2),$
           origspec:ptr_new(2),$
           naps:0L,$
           norders:0L,$           
           packagepath:packagepath,$
           path:'',$
           pixmap_wid:0L,$
           pixpp:250.0,$
           plotatmosphere:0,$
           plotlincor:1L,$
           plotfixed:0L,$
           plotwin_wid:0L,$
           plotwin:0L,$
           plotwin_size:[screensize[0]*0.5,screensize[1]*0.6],$
           pscale:!p,$
           reg:[[!values.d_nan,!values.d_nan],$
                [!values.d_nan,!values.d_nan]],$
           selectord_dl:0L,$
           sg_base:0L,$
           sgdeg_fld:[0L,0L],$
           sgwin_fld:[0L,0L],$           
           slider:0L,$
           sliderval:50,$
           smthhistory:'',$
           smoothtype:'Savitzky-Golay',$
           snr_fld:[0L,0L],$
           snrcut:0.0,$
           snrxrange:[0.,0.],$
           snryrange:[0.,0.],$
           snryranges:ptr_new(2),$
           spectype:'Flux',$
           spextoolpath:spextoolpath,$
           uncxrange:[0.,0.],$
           uncyrange:[0.,0.],$
           uncyranges:ptr_new(2),$
           edit_base:0L,$
           xmax_fld:[0L,0L],$
           xmin_fld:[0L,0L],$
           xrange:[0.,0.],$
           xranges:ptr_new(2),$
           xscale:!x,$
           xtitle:'',$
           ymax_fld:[0L,0L],$
           ymin_fld:[0L,0L],$
           yranges:ptr_new(2),$
           yrange:[0.,0.],$
           yscale:!y,$
           ytitles:['','',''],$
           ytitle:'',$
           winbuffer:[0L,0L],$
           weighted:1,$
           workfixmask:ptr_new(2),$
           workrmvmask:ptr_new(2),$
           workspec:ptr_new(2),$
           xcleanspec_base:0L}

;  Create the widget
  
  title = 'xcleanspec '+version+' for '+state.instrument
  
  state.xcleanspec_base = widget_base(TITLE=title, $
                                        /COLUMN,$
                                        /TLB_SIZE_EVENTS)
  
     quit_button = widget_button(state.xcleanspec_base,$
                                 FONT=buttonfont,$
                                 EVENT_PRO='xcleanspec_event',$
                                 VALUE='Quit',$
                                 UVALUE='Quit')
     
     row_base = widget_base(state.xcleanspec_base,$
                            /ROW)

        col1_base = widget_base(row_base,$
                                EVENT_PRO='xcleanspec_event',$
                                /COLUMN)
        
           box1_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box1_base,$
                                   VALUE='1.  Load Spectra',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)

              
              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 input = widget_button(row,$
                                       FONT=buttonfont,$
                                       VALUE='File Name',$
                                       UVALUE='Input Spectrum',$
                                       EVENT_PRO='xcleanspec_event')
                 
               
                 input_fld = coyote_field2(row,$
                                           LABELFONT=buttonfont,$
                                           FIELDFONT=textfont,$
                                           TITLE=':',$
                                           UVALUE='Input Spectrum Field',$
                                           XSIZE=20,$
                                           EVENT_PRO='xcleanspec_event',$
                                           /CR_ONLY,$
                                           TEXTID=textid)
                 state.ispectrum_fld = [input_fld,textid]
              
              load = widget_button(box1_base,$
                                   VALUE='Load Spectra',$
                                   UVALUE='Load Spectra',$
                                   FONT=buttonfont)

           box2_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)

              label = widget_label(box2_base,$
                                   VALUE='2.  Choose Aperture',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)

              state.ap_dl = widget_droplist(box2_base,$
                                            FONT=buttonfont,$
                                            TITLE='Aperture: ',$
                                            VALUE='01',$
                                            UVALUE='Aperture')
              
           box3_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box3_base,$
                                   VALUE='3.  S/N Cut',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)

              fld = coyote_field2(box3_base,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE='S/N Limit (0=all data):',$
                                  UVALUE='S/N Cut Field',$
                                  VALUE=state.snrcut,$
                                  XSIZE=6,$
                                  EVENT_PRO='xcleanspec_event',$
                                  /CR_ONLY,$
                                  TEXTID=textid)
              state.snr_fld = [fld,textid]

           box4_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)

              label = widget_label(box4_base,$
                                   VALUE='4.  Clean Spectra',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)           
                            
              state.selectord_dl = widget_droplist(box4_base,$
                                                   FONT=buttonfont,$
                                                   TITLE='Select Order: ',$
                                                   VALUE='01',$
                                                   UVALUE='Select Order Button')

              state.edit_base = widget_base(box4_base,$
                                            /ROW,$
                                            /BASE_ALIGN_CENTER)

                 row = widget_base(state.edit_base,$
                                   /ROW,$
                                   /TOOLBAR,$
                                   /EXCLUSIVE)


                    state.mbut_fix = widget_button(row, $
                                                   VALUE='Fix', $
                                                   /NO_RELEASE,$
                                                EVENT_PRO='xcleanspec_event',$
                                                   UVALUE='Fix Button',$
                                                   FONT=buttonfont)
                 
                    state.mbut_rmv = widget_button(row, $
                                                   VALUE='Remove', $
                                                   /NO_RELEASE,$
                                                EVENT_PRO='xcleanspec_event',$
                                                   UVALUE='Remove Button',$
                                                   FONT=buttonfont)

                 state.mbut_undo = widget_button(state.edit_base, $
                                                 VALUE='Undo', $
                                                 /NO_RELEASE,$
                                                 EVENT_PRO='xcleanspec_event',$
                                                 UVALUE='Undo Button',$
                                                 FONT=buttonfont)
                    
                 state.mbut_save = widget_button(state.edit_base, $
                                                 VALUE='Save', $
                                                 /NO_RELEASE,$
                                                 EVENT_PRO='xcleanspec_event',$
                                                 UVALUE='Save Button',$
                                                 FONT=buttonfont)

           state.box5_base = widget_base(col1_base,$
                                         /COLUMN,$
                                         FRAME=2)
           
              label = widget_label(state.box5_base,$
                                   VALUE='5.  Smooth Spectra',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)
              
              smooth_bg = cw_bgroup(state.box5_base,$
                                    FONT=buttonfont,$
                                    ['Savitzky-Golay','Gaussian'],$
                                    /ROW,$
                                    /RETURN_NAME,$
                                    /NO_RELEASE,$
                                    /EXCLUSIVE,$
                                    LABEL_LEFT='',$
                                    UVALUE='Smooth Type',$
                                    SET_VALUE=0)
              
              row = widget_base(state.box5_base)

                 state.sg_base = widget_base(row,$
                                             /ROW,$
                                             /BASE_ALIGN_CENTER)
                 
                    window = coyote_field2(state.sg_base,$
                                           LABELFONT=buttonfont,$
                                           FIELDFONT=textfont,$
                                           TITLE='Width (pixels):',$
                                           UVALUE='SG Width',$
                                           XSIZE=3,$
                                           TEXTID=textid)
                    state.sgwin_fld = [window,textid] 
                    
                    window = coyote_field2(state.sg_base,$
                                           LABELFONT=buttonfont,$
                                           FIELDFONT=textfont,$
                                           TITLE='Degree:',$
                                           UVALUE='SG Degree',$
                                           VALUE='2',$
                                           XSIZE=2,$
                                           TEXTID=textid)
                    state.sgdeg_fld = [window,textid] 
                    
                 state.gs_base = widget_base(row,$
                                             /ROW,$
                                             /BASE_ALIGN_CENTER,$
                                             MAP=0)
                 
                    field = coyote_field2(state.gs_base,$
                                          LABELFONT=buttonfont,$
                                          FIELDFONT=textfont,$
                                          TITLE='FWHM (pixels)=',$
                                          UVALUE='FWHM',$
                                          XSIZE=5,$
                                          TEXTID=textid)
                    state.fwhm_fld = [field,textid] 
                    
              smooth_button = widget_button(state.box5_base,$
                                            FONT=buttonfont,$
                                            VALUE='Smooth Spectra',$
                                            UVALUE='Smooth Spectra')
                 
           box6_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box6_base,$
                                   VALUE='6.  Write File',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)

              oname = coyote_field2(box6_base,$
                                    LABELFONT=buttonfont,$
                                    FIELDFONT=textfont,$
                                    TITLE='File Name:',$
                                    UVALUE='Object File Oname',$
                                    XSIZE=20,$
                                    TEXTID=textid)
            state.oname_fld = [oname,textid]
            
            write = widget_button(box6_base,$
                                  VALUE='Write File',$
                                  UVALUE='Write File',$
                                  FONT=buttonfont)
              

              
        col2_base = widget_base(row_base,$
                                FRAME=2,$
                                /COLUMN)

           row = widget_base(col2_base,$
                             /ROW,$
                             /BASE_ALIGN_CENTER,$
                             EVENT_PRO='xcleanspec_event')

           widget_label = widget_label(row,$
                                       VALUE='Plot:',$
                                       FONT=buttonfont)

              subrow = widget_base(row,$
                                   /ROW,$
                                   /TOOLBAR,$
                                   /EXCLUSIVE)
              
                 state.mbut_flx = widget_button(subrow, $
                                                VALUE='Flux', $
                                                /NO_RELEASE,$
                                                EVENT_PRO='xcleanspec_event',$
                                                UVALUE='Plot Flux Button',$
                                                FONT=buttonfont)
                 
                 state.mbut_unc = widget_button(subrow, $
                                                VALUE='Uncertainty', $
                                                /NO_RELEASE,$
                                                EVENT_PRO='xcleanspec_event',$
                                             UVALUE='Plot Uncertainty Button',$
                                                FONT=buttonfont)
                 
                 state.mbut_snr = widget_button(subrow, $
                                                VALUE='S/N', $
                                                /NO_RELEASE,$
                                                EVENT_PRO='xcleanspec_event',$
                                                UVALUE='Plot S/N Button',$
                                                FONT=buttonfont)
                 widget_control, state.mbut_flx,/SET_BUTTON

              label = widget_label(row,$
                                   VALUE=' ')
                 
              subrow = widget_base(row,$
                                   /ROW,$
                                   /TOOLBAR,$
                                   /NONEXCLUSIVE)
              
                 state.mbut_atmos = widget_button(subrow, $
                                                  VALUE='Atmosphere', $
                                                  EVENT_PRO='xcleanspec_event',$
                                              UVALUE='Plot Atmosphere Button',$
                                                  FONT=buttonfont)
                 widget_control, state.mbut_atmos, $
                                 SET_BUTTON=state.plotatmosphere

                 subrow = widget_base(row,$
                                      /ROW,$
                                      /TOOLBAR,$
                                      /EXCLUSIVE)
                 
                    button = widget_button(subrow, $
                                           VALUE='2 Color', $
                                           EVENT_PRO='xcleanspec_event',$
                                           UVALUE='2 Color Button',$
                                           FONT=state.buttonfont)
                    if state.altcolor eq 2 then widget_control, button, $
                       /SET_BUTTON
              
                    button = widget_button(subrow, $
                                           VALUE='3 Color', $
                                           EVENT_PRO='xcleanspec_event',$
                                           UVALUE='3 Color Button',$
                                           FONT=state.buttonfont)
                    if state.altcolor eq 3 then widget_control, button, $
                       /SET_BUTTON
                 
              label = widget_label(row,$
                                   VALUE=' ')
                 
              subrow = widget_base(row,$
                                   /ROW,$
                                   /TOOLBAR,$
                                   /NONEXCLUSIVE)
              
                 state.mbut_lincor = widget_button(subrow, $
                                                   VALUE='Lincor Flags', $
                                                EVENT_PRO='xcleanspec_event',$
                                                   UVALUE='Plot Lincor Button',$
                                                   FONT=buttonfont)
                 widget_control, state.mbut_lincor,SET_BUTTON=state.plotlincor
                 
                 state.mbut_fixed = widget_button(subrow, $
                                                  VALUE='Fixed Flags', $
                                                  EVENT_PRO='xcleanspec_event',$
                                                  UVALUE='Plot Fixed Button',$
                                                  FONT=buttonfont)
                 widget_control, state.mbut_fixed,SET_BUTTON=state.plotfixed
                                           
           row = widget_base(col2_base,$
                             /ROW,$
                             /BASE_ALIGN_CENTER)
           
              state.plotwin = widget_draw(row,$
                                          /ALIGN_CENTER,$
                                          XSIZE=state.plotwin_size[0],$
                                          YSIZE=state.plotwin_size[1],$
                                       EVENT_PRO='xcleanspec_plotwinevent',$
                                          /KEYBOARD_EVENTS,$
                                          /BUTTON_EVENTS,$
                                          /TRACKING_EVENTS,$
                                          /MOTION_EVENTS)
              
           state.slider = widget_slider(col2_base,$
                                        UVALUE='Slider',$
                                        EVENT_PRO='xcleanspec_event',$
                                        /DRAG,$
                                        /SUPPRESS_VALUE,$
                                        FONT=buttonfont)
           widget_control, state.slider, SET_VALUE=state.sliderval
              
           row_base = widget_base(col2_base,$
                                  /ROW)
           
           xmin = coyote_field2(row_base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='X Min:',$
                                UVALUE='X Min',$
                                XSIZE=15,$
                                EVENT_PRO='xcleanspec_minmaxevent',$
                                /CR_ONLY,$
                                TEXTID=textid)
           state.xmin_fld = [xmin,textid]
           
           xmax = coyote_field2(row_base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='X Max:',$
                                UVALUE='X Max',$
                                XSIZE=15,$
                                EVENT_PRO='xcleanspec_minmaxevent',$
                                /CR_ONLY,$
                                TEXTID=textid)
           state.xmax_fld = [xmax,textid]
           
           ymin = coyote_field2(row_base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Y Min:',$
                                UVALUE='Y Min',$
                                XSIZE=15,$
                                EVENT_PRO='xcleanspec_minmaxevent',$
                                /CR_ONLY,$
                                TEXTID=textid)
           state.ymin_fld = [ymin,textid]
           
           ymax = coyote_field2(row_base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Y Max:',$
                                UVALUE='Y Max',$
                                XSIZE=15,$
                                EVENT_PRO='xcleanspec_minmaxevent',$
                                /CR_ONLY,$
                                TEXTID=textid)
           state.ymax_fld = [ymax,textid]
           
   button = widget_button(state.xcleanspec_base,$
                          FONT=buttonfont,$
                          EVENT_PRO='xcleanspec_event',$
                          VALUE='Help',$
                          UVALUE='Help')
           
; Get things running.  Center the widget using the Fanning routine.
      
   cgcentertlb,state.xcleanspec_base
   
   widget_control, state.xcleanspec_base, /REALIZE

;  Get plotwin ids

   widget_control, state.plotwin, GET_VALUE=x
   state.plotwin_wid=x
   window, /FREE, /PIXMAP,XSIZE=state.plotwin_size[0], $
           YSIZE=state.plotwin_size[1]
   state.pixmap_wid = !d.window

   erase, COLOR=20
   wset, state.plotwin_wid
   device, COPY=[0,0,state.plotwin_size[0],state.plotwin_size[1],0,0,$
                 state.pixmap_wid]
   
;  Get sizes for things.
   
   widget_geom = widget_info(state.xcleanspec_base, /GEOMETRY)
   
   state.winbuffer[0]=widget_geom.xsize-state.plotwin_size[0]
   state.winbuffer[1]=widget_geom.ysize-state.plotwin_size[1]
    
; Start the Event Loop. 
    
   XManager, 'xcleanspec', $
             state.xcleanspec_base, $
             CLEANUP='xcleanspec_cleanup',$
             EVENT_HANDLER='xcleanspec_resizeevent',$
             /NO_BLOCK
      
end
