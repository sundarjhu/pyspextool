pro xmergeorders_initcommon

  common xmergeorders_state, state

end
;
;==============================================================================
;
pro xmergeorders_chooseaperture

  common xmergeorders_state


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
        state.ytitle = state.ytitle[2]
        *state.yranges = *state.snryranges
        
     end
     
  endcase

  state.absxrange = state.xrange
  state.absyrange = state.yrange
    
  state.reg = !values.f_nan
  state.cliporder = -1
 
end
;
;==============================================================================
;
pro xmergeorders_cleanup,base

  common xmergeorders_state
  
  if n_elements(state) ne 0 then begin

  ptr_free, state.origspec  
  ptr_free, state.hdr     
  ptr_free, state.orders 
  ptr_free, state.xranges
  ptr_free, state.yranges
  ptr_free, state.workspec
  ptr_free, state.snrmask
  ptr_free, state.usrmask
  ptr_free, state.cenwave
  ptr_free, state.mergedspec
  ptr_free, state.awave
  ptr_free, state.atrans
  ptr_free, state.snlimit
  
  ptr_free, state.uncyranges
  ptr_free, state.flxyranges
  ptr_free, state.snryranges
  
endif
  
  state = 0B

end
;
;=============================================================================
;
pro xmergeorders_loadspec

  common xmergeorders_state

;  Get files.

  file = mc_cfld(state.ispectrum_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  file = mc_cfile(file,WIDGET_ID=state.xmergeorders_base,CANCEL=cancel)
  if cancel then return

;  Copy root of file to output field

  widget_control, state.oname_fld[1], $
                  SET_VALUE='m'+strtrim(file_basename(file,'.fits'),2)
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
  *state.snlimit  = replicate(0,2)
  
  npixels = n_elements((*state.origspec)[*,0,0])
  *state.snrmask = make_array(npixels,state.norders,state.naps,/BYTE,VALUE=1)
  *state.usrmask = make_array(npixels,state.norders,state.naps,/BYTE,VALUE=1)
  
;  Deal with the units
  
  state.xtitle = xtitle
  lidx = strpos(ytitle,'(')
  ridx = strpos(ytitle,')')
  ypunits = strmid(ytitle,lidx+1,ridx-lidx-1)
  state.ytitles = [ytitle,'!5Uncertainty ('+ypunits+')','!5S/N']
  
;  Update droplists
  
  widget_control,state.ap_dl,$
                 SET_VALUE=strtrim(string(indgen(naps)+1,FORMAT='(I2)'),2)
  widget_control, state.ap_dl, SENSITIVE=(state.naps eq 1) ? 0:1

  val = ['None',strtrim(string(orders,FORMAT='(I3)'),2)]
  widget_control,state.selectord_dl,SET_VALUE=val  

  widget_control, state.snlimit_fld[1],SET_VALUE='0'
  
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
  xmergeorders_chooseaperture
  xmergeorders_mergeorders
  xmergeorders_setminmax
  xmergeorders_plotspec

;  Unfreeze the widget
  
  state.freeze = 0
  
end
;
;=============================================================================
;
pro xmergeorders_mergeorders

  common xmergeorders_state

  for j = 0,state.naps-1 do begin

     merged_w = (*state.workspec)[*,0,j]
     merged_f = (*state.workspec)[*,1,j]
     merged_u = (*state.workspec)[*,2,j]
     merged_b = (*state.workspec)[*,3,j]
     
     zsnr = where((*state.snrmask)[*,0,state.ap] eq 0,cntsnr)
     zusr = where((*state.usrmask)[*,0,state.ap] eq 0,cntusr)
     
     if cntsnr ne 0 then merged_f[zsnr] = !values.f_nan
     if cntusr ne 0 then merged_f[zusr] = !values.f_nan
          
     for i = 1,state.norders-1 do begin

        tmp_w = (*state.workspec)[*,0,i*state.naps+j]
        tmp_f = (*state.workspec)[*,1,i*state.naps+j]
        tmp_u = (*state.workspec)[*,2,i*state.naps+j]
        tmp_b = (*state.workspec)[*,3,i*state.naps+j]
        
        zsnr = where((*state.snrmask)[*,i] eq 0,cntsnr)
        zusr = where((*state.usrmask)[*,i] eq 0,cntusr)
        
        if cntsnr ne 0 then tmp_f[zsnr] = !values.f_nan
        if cntusr ne 0 then tmp_f[zusr] = !values.f_nan
               
        mc_mergespec,merged_w, double(merged_f), tmp_w, double(tmp_f), $
                     owave,oflux,E1=double(merged_u),E2=double(tmp_u), $
                     BF1=merged_b,BF2=tmp_b,OERROR=oerror,OBITFLAG=obitflag, $
                     CANCEL=cancel
        if cancel then return
        
        merged_w = owave
        merged_f = oflux
        merged_u = oerror
        merged_b = obitflag        
      
     endfor
     
     arr = [[merged_w],[merged_f],[merged_u],[merged_b]]
     *state.mergedspec = (j eq 0) ? arr:[[[*state.mergedspec]],[[arr]]]

  endfor

end
;
;===============================================================================
;
pro xmergeorders_pickspectra,current,new

  common xmergeorders_state
  
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
pro xmergeorders_plotspec

  common xmergeorders_state

  wset, state.pixmap_wid
  erase,COLOR=20
  
;  Get plot position for overlapping spectral plot (to leave space for
;  the masks and order numbers
  
  if state.plottype eq 'Overlapped' then begin

     position = [120,60,state.plotwin_size[0]-20,state.plotwin_size[1]-100]

  endif else begin

     position = [120,60,state.plotwin_size[0]-20,state.plotwin_size[1]-40]     
     title = 'Merged Spectrum'

  endelse

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

  plot,[1],[1],/XSTYLE,XTITLE=state.xtitle,YTITLE=state.ytitle, $
       CHARSIZE=state.charsize,/NODATA,YRANGE=state.yrange, $
       XRANGE=state.xrange,YSTYLE=ystyle,BACKGROUND=20,NOERASE=noerase, $
       POSITION=position,/DEVICE,TITLE=title
    
;  Do the order by order plotting 
  
  if state.plottype eq 'Overlapped' then begin

     spectra = *state.workspec

     xyouts,20,state.plotwin_size[1]-20,'!5Order',ALIGNMENT=0,/DEVICE, $
            CHARSIZE=state.charsize
     xyouts,20,state.plotwin_size[1]-55,'!5S/N Mask',ALIGNMENT=0,/DEVICE, $
            CHARSIZE=state.charsize
     xyouts,20,state.plotwin_size[1]-80,'!5User Mask',ALIGNMENT=0,/DEVICE, $
            CHARSIZE=state.charsize

     l = 0
     for i = 0, state.norders-1 do begin

        idx = i*state.naps+state.ap        
        if ~mc_rangeinrange((*state.xranges)[*,i],!x.crange) then continue
     
        zsnr = where((*state.snrmask)[*,i,state.ap] eq 0,cntsnr)
        zusr = where((*state.usrmask)[*,i,state.ap] eq 0,cntusr)
        
        if cntsnr ne 0 then spectra[zsnr,1:2,idx] = !values.f_nan
        if cntusr ne 0 then spectra[zusr,1:2,idx] = !values.f_nan
        
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
        
;        color = (i mod 2) ? 1:3

        if state.cliporder ne -1 then begin

           if i eq state.cliporder then begin

              savewave = wave
              saveflux = spec
              savecolor = color
              
           endif
           oplot,wave,spec,COLOR=100,PSYM=10

        endif else oplot,wave,spec,COLOR=color,PSYM=10

;  label orders

        min = max([!x.crange[0],(*state.xranges)[0,i]])
        max = min([!x.crange[1],(*state.xranges)[1,i]])
        
        lwave = (min+max)/2.

        if lwave gt !x.crange[0] then begin

           xy = convert_coord(lwave,!y.crange[1],/DATA,/TO_DEVICE)
                                          
           xyouts,xy[0],state.plotwin_size[1]-15-15*(i mod 2), $
                  strtrim(string((*state.orders)[i],FORMAT='(I3)'),2), $
                  /DEVICE,COLOR=color,ALIGNMENT=0.5,CHARSIZE=state.charsize


        endif
               
;  Plot masks
        
        if cntsnr ne 0 then begin

           xy = convert_coord(spectra[zsnr,0,idx],!y.crange[1],/DATA,/TO_DEVICE)

           plots,reform(xy[0,*]),state.plotwin_size[1]-50,PSYM=1,COLOR=color, $
                 /DEVICE
           
        endif
        
        if cntusr ne 0 then begin
           
           xy = convert_coord(spectra[zusr,0,idx],!y.crange[1],/DATA,/TO_DEVICE)
           plots,reform(xy[0,*]),state.plotwin_size[1]-70,PSYM=1,COLOR=color, $
                 /DEVICE
           
        endif
        l = l + 1
        
     endfor

     if state.cliporder ne -1 then oplot,savewave,saveflux, $
                                         COLOR=savecolor,PSYM=10        
     
  endif

  if state.plottype eq 'Merged' then begin

     spectra = *state.mergedspec

     case state.spectype of 
        
        'Flux': spec = spectra[*,1,state.ap]
        
        'Uncertainty': spec = spectra[*,2,state.ap]
       
        'S/N': spec = spectra[*,1]/spectra[*,2,state.ap]
        
     endcase
     oplot,(*state.mergedspec)[*,0,state.ap],spec,PSYM=10

  endif

  if ystyle eq 9 then begin

     ticks = ['0.0','0.2','0.4','0.6','0.8','1.0']
     axis,YAXIS=1,YTICKS=5,YTICKNAME=ticks,YMINOR=2,COLOR=5, $
          CHARSIZE=state.charsize
     
  endif
  
  if !y.crange[0] lt 0 then plots,!x.crange,[0,0],LINESTYLE=1
  
  state.xscale = !x
  state.yscale = !y
  state.pscale = !p
  
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plotwin_size[0],state.plotwin_size[1],0,0, $
                state.pixmap_wid]
  
end
;
;=============================================================================
;
pro xmergeorders_setminmax

  common xmergeorders_state
  
  widget_control, state.xmin_fld[1], SET_VALUE=strtrim(state.xrange[0],2)
  widget_control, state.xmax_fld[1], SET_VALUE=strtrim(state.xrange[1],2)

  widget_control, state.ymin_fld[1], SET_VALUE=strtrim(state.yrange[0],2)
  widget_control, state.ymax_fld[1], SET_VALUE=strtrim(state.yrange[1],2)

  xmergeorders_setslider

end
;
;=============================================================================
;
pro xmergeorders_setslider

  common xmergeorders_state

;  Get new slider value
  
  del = state.absxrange[1]-state.absxrange[0]
  midwave = (state.xrange[1]+state.xrange[0])/2.
  state.sliderval = round((midwave-state.absxrange[0])/del*100)
     
  widget_control, state.slider, SET_VALUE=state.sliderval
  
end
;
;=============================================================================
;
pro xmergeorders_snmask

  common xmergeorders_state

  lim = mc_cfld(state.snlimit_fld,2,CANCEL=cancel)
  if cancel then return

  (*state.snlimit)[state.ap] = lim
  (*state.snrmask)[*] = 1
  
  if lim ne '' then begin

     for i = 0,state.norders-1 do begin
        
        snr = (*state.workspec)[*,1,i]/(*state.workspec)[*,2,i]
        z = where(snr lt lim,cnt,NCOMP=ncomp)
        if cnt ne 0 then (*state.snrmask)[z,i,state.ap] = 0
        
     endfor

  endif

end
;
;===============================================================================
;
;
pro xmergeorders_writefile

  common xmergeorders_state
  
  ifile = mc_cfld(state.ispectrum_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return

  file = mc_cfld(state.oname_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return

  hdr = *state.hdr
  
  fxaddpar,hdr,'FILENAME',file
  fxaddpar,hdr,'CREPROG','xtellcor', ' Creation IDL program'  
  fxaddpar,hdr,'ORDERS','0', ' Order numbers'
  fxaddpar,hdr,'NORDERS',1, ' Number of orders'
  fxaddpar,hdr,'NAPS',state.naps, ' Number of apertures'

  history = 'This merged spectrum was derived from the file '+ $
            strtrim(file_basename(ifile),2)+'.'

  if total(*state.snlimit) ne 0 then begin

     if state.naps gt 1 then begin
     
        history = history+'  S/N cuts of '+ $
                  strjoin(strtrim(string(*state.snlimit,FORMAT='(I4)'),2),',')+$
                  ' were applied to the apertures before merging.'

     endif else begin

        history = history+'  A S/N cut of '+ $
                  strtrim(string((*state.snlimit)[0],FORMAT='(I4)'),2)+$
                  ' was applied to the spectrum merging.'        

     endelse
        
  endif

  sxaddhist,' ',hdr    
  sxaddhist,'###################### Xmergeorders History ' + $
            '#######################',hdr
  sxaddhist,' ',hdr

  print, history
  history = mc_splittext(history,67,CANCEL=cancel)
  if cancel then return
  sxaddhist,history,hdr
  
  writefits,state.path+file+'.fits',*state.mergedspec,hdr
  xvspec,state.path+file+'.fits'
  
  end
;
;===============================================================================
;
pro xmergeorders_zoom,IN=in,OUT=out

  common xmergeorders_state

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

    xmergeorders_plotspec
    xmergeorders_setminmax

    
end
;
;=============================================================================
;
;----------------------------- Event Handlder -------------------------------
;
;=============================================================================
;
pro xmergeorders_event, event

  common xmergeorders_state
  
;  Check to see if it is the help file 'Done' Button
  
  widget_control, event.id,  GET_UVALUE = uvalue
  
;  Must be a main widget event
  
  case uvalue of

     '2 Color Button': begin

        state.altcolor = 2
        if ~state.freeze then xmergeorders_plotspec
        
     end

     '3 Color Button': begin

        state.altcolor = 3
        if ~state.freeze then xmergeorders_plotspec
        
     end

     'Aperture': begin

        state.ap = event.index
        widget_control, state.snlimit_fld[1], $
                        SET_VALUE=strtrim(string((*state.snlimit)[state.ap], $
                                         FORMAT='(I4)'),2)
        state.cliporder = -1
        widget_control, state.selectord_dl,SET_DROPLIST_SELECT=0
        state.cursormode = 'None'
        xmergeorders_chooseaperture
        xmergeorders_setminmax
        xmergeorders_plotspec

     end
     
     'Help':  begin

        pre = (strupcase(!version.os_family) eq 'WINDOWS') ? 'start ':'open '
        
        spawn, pre+filepath(strlowcase(state.instrument)+'_spextoolmanual.pdf',$
                            ROOT=state.packagepath,$
                            SUBDIR='manual')
        
     end
     
     'Input Spectrum': begin
        
        obj = dialog_pickfile(DIALOG_PARENT=state.xmergeorders_base,$
                              PATH=state.path, GET_PATH=path,$
                              /MUST_EXIST,FILTER='*.fits')
        if obj ne '' then begin
           
           widget_control,state.ispectrum_fld[1],SET_VALUE = strtrim(obj,2)
           mc_setfocus,state.ispectrum_fld
           state.path = path

        endif
           
     end

     'Load Spectra': xmergeorders_loadspec
        
     'Plot Atmosphere Button': begin

        state.plotatmosphere = event.select
        if state.freeze then return
        xmergeorders_plotspec

     end
     
     'Plot Merged Button': begin

        state.plottype = 'Merged'
        if state.freeze then return
        xmergeorders_plotspec

     end
     
     'Plot Flux Button': begin

        if state.freeze then return
        xmergeorders_pickspectra,state.spectype,'Flux'        
        xmergeorders_setminmax
        xmergeorders_plotspec

     end

     'Plot Overlapped Button': begin

        state.plottype = 'Overlapped'
        if state.freeze then return
        xmergeorders_plotspec

     end
     
     'Plot S/N Button': begin

        if state.freeze then return
        xmergeorders_pickspectra,state.spectype,'S/N'        
        xmergeorders_setminmax
        xmergeorders_plotspec

     end

     'Plot Uncertainty Button': begin

        if state.freeze then return
        xmergeorders_pickspectra,state.spectype,'Uncertainty'        
        xmergeorders_setminmax
        xmergeorders_plotspec
        
     end

     'Quit': begin

        widget_control, event.top, /DESTROY
        return
        
     end
     
     'Select Order Button': begin

        if state.freeze then return
        state.cliporder = (event.index eq 0) ? -1:(event.index-1)
        if state.cliporder eq -1 then state.cursormode = 'None'
        xmergeorders_plotspec
                  
     end
     
     'Slider': begin

        if state.freeze then return
        del = state.absxrange[1]-state.absxrange[0]
        oldcen = (state.xrange[1]+state.xrange[0])/2.
        newcen = state.absxrange[0]+del*(event.value/100.)
        
        
        state.xrange = state.xrange + (newcen-oldcen)
        xmergeorders_plotspec
        
        if event.drag eq 0 then xmergeorders_setminmax
        
     end

     'S/N Limit Field': begin

        if state.freeze then return        
        xmergeorders_snmask
        xmergeorders_mergeorders
        xmergeorders_plotspec

     end

     'Write File': begin
        
        if state.freeze then return
        xmergeorders_writefile
        
     end

     else:
     
  endcase
  
end
;
;===============================================================================
;
pro xmergeorders_minmaxevent,event

  common xmergeorders_state

  widget_control, event.id,  GET_UVALUE = uvalue
  
  if state.freeze then goto, cont

  case uvalue of 
     
     'X Min': begin
        
        xmin = mc_cfld(state.xmin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then goto, cont
        xmin = mc_crange(xmin,state.xrange[1],'X Min',/KLT,$
                         WIDGET_ID=state.xmergeorders_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control, state.xmin_fld[0],SET_VALUE=state.xrange[0]
           goto, cont
           
        endif else state.xrange[0] = xmin
        
     end
     
     'X Max': begin
        
        xmax = mc_cfld(state.xmax_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then goto, cont
        xmax = mc_crange(xmax,state.xrange[0],'X Max',/KGT,$
                       WIDGET_ID=state.xmergeorders_base,CANCEL=cancel)
        if cancel then begin
            
            widget_control, state.xmax_fld[0],SET_VALUE=state.xrange[1]
            goto, cont
            
        endif else state.xrange[1] = xmax

    end

    'Y Min': begin

        ymin = mc_cfld(state.ymin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then goto, cont
        ymin = mc_crange(ymin,state.yrange[1],'Y Min',/KLT,$
                         WIDGET_ID=state.xmergeorders_base,CANCEL=cancel)
        if cancel then begin
           
            widget_control,state.ymin_fld[0],SET_VALUE=state.yrange[0]
            goto, cont
            
        endif else state.yrange[0] = ymin
        
    end
     
    'Y Max': begin
       
       ymax = mc_cfld(state.ymax_fld,4,/EMPTY,CANCEL=cancel)
       if cancel then return
       ymax = mc_crange(ymax,state.yrange[0],'Y Max',/KGT,$
                        WIDGET_ID=state.xmergeorders_base,CANCEL=cancel)
       if cancel then begin
          
          widget_control,state.ymax_fld[0],SET_VALUE=state.yrange[1]
          goto, cont
          
        endif else state.yrange[1] = ymax
        
    end

    
 endcase

;  Set slider and update plot

  xmergeorders_setslider
  xmergeorders_plotspec

cont:

end
;
;===============================================================================
;
pro xmergeorders_plotwinevent,event

  common xmergeorders_state

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

     case event.key of

        5: begin ; left

           if state.freeze then return           

           del = (state.xrange[1]-state.xrange[0])*0.3
           oldcen = (state.xrange[1]+state.xrange[0])/2.
           newcen = oldcen-del
           
           if newcen lt state.absxrange[0] then return
           state.xrange = state.xrange + (newcen-oldcen)
           xmergeorders_setminmax
           xmergeorders_plotspec

           
        end

        6: begin ; right

           if state.freeze then return           

           del = (state.xrange[1]-state.xrange[0])*0.3
           oldcen = (state.xrange[1]+state.xrange[0])/2.
           newcen = oldcen+del
           
           if newcen gt state.absxrange[1] then return
           state.xrange = state.xrange + (newcen-oldcen)
           xmergeorders_setminmax
           xmergeorders_plotspec
           
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
           state.reg = !values.f_nan                
           xmergeorders_plotspec
           
        end

        'd': begin

           state.cliporder = -1
           widget_control, state.selectord_dl,SET_DROPLIST_SELECT=0
           state.cursormode = 'None'
           xmergeorders_plotspec

        end
        
        'i': xmergeorders_zoom,/IN

        'm': begin

        end
        
        'o': xmergeorders_zoom,/OUT

        'q': begin

           widget_control, event.top, /DESTROY
           return

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

              state.xrange = mc_bufrange((*state.xranges)[*,z],0.05)
              state.yrange = mc_bufrange((*state.yranges)[*,z],0.05)
              xmergeorders_setminmax
              xmergeorders_plotspec                
              
           endif else begin

              state.cursormode = 'Mask'
              state.reg = !values.f_nan                
              xmergeorders_plotspec
                         
           endelse
                     
        end

        'u': begin

           if state.cliporder eq -1 then  begin

              message = dialog_message('Please select an order first.',/ERROR,$
                                    DIALOG_PARENT=state.xmergeorders_base, $
                                       /CENTER)
              return

           endif
           
           state.cursormode = 'Undo'
           state.reg = !values.f_nan                
           xmergeorders_plotspec

        end
        
        'w': begin
           
           state.xrange = state.absxrange
           state.yrange = state.absyrange
           xmergeorders_setminmax
           xmergeorders_plotspec
           
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
           
           'XZoom': plots, [event.x,event.x],$
                           [0,state.plotwin_size[1]],COLOR=2,/DEVICE, $
                           LINESTYLE=2
           
           'YZoom': plots, [0,state.plotwin_size[0]],$
                           [event.y,event.y],COLOR=2,/DEVICE,LINESTYLE=2

           'Mask': begin

              plots,[state.reg[0,0],state.reg[0,0]],!y.crange,COLOR=7,THICK=2,$
                    LINESTYLE=2
              
           end

           'Undo': begin

              plots,[state.reg[0,0],state.reg[0,0]],!y.crange,COLOR=7,THICK=2,$
                    LINESTYLE=2
              
           end
           
           else:
           
        endcase
        wset, state.plotwin_wid
        device, COPY=[0,0,state.plotwin_size[0], $
                      state.plotwin_size[1],0,0,state.pixmap_wid]
        
     endif else begin 
        
        state.reg[*,1] = xy[0:1]
        case state.cursormode of 
           
           'XZoom': state.xrange = [min(state.reg[0,*],MAX=max),max]
           
           'YZoom': state.yrange = [min(state.reg[1,*],MAX=max),max]
           
           'Zoom': begin
              
              state.xrange = [min(state.reg[0,*],MAX=max),max]
              state.yrange = [min(state.reg[1,*],MAX=max),max]
              
           end

           'Mask':  begin

              min = min(state.reg[0,*],MAX=max)
              idx = state.cliporder*state.naps+state.ap
              z = where((*state.workspec)[*,0,idx] gt min and $
                        (*state.workspec)[*,0,idx] lt max,cnt)
              if cnt ne 0 then (*state.usrmask)[z,state.cliporder,state.ap] = 0
              xmergeorders_mergeorders
              xmergeorders_plotspec
              
           end           

           'Undo': begin

              min = min(state.reg[0,*],MAX=max)
              idx = state.cliporder*state.naps+state.ap
              z = where((*state.workspec)[*,0,idx] gt min and $
                        (*state.workspec)[*,0,idx] lt max and $
                        (*state.usrmask)[*,state.cliporder,state.ap] eq 0,cnt)
              if cnt ne 0 then (*state.usrmask)[z,state.cliporder,state.ap] = 1
              xmergeorders_mergeorders
              xmergeorders_plotspec
              
           end
           
           else:
           
        endcase

        xmergeorders_setminmax
        xmergeorders_plotspec
        state.cursormode='None'
        
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

     'Mask': plots, [event.x,event.x],[0,state.plotwin_size[1]],COLOR=7,/DEVICE

     'Undo': plots, [event.x,event.x],[0,state.plotwin_size[1]],COLOR=7,/DEVICE
     
     else: begin
        
        plots, [event.x,event.x],[0,state.plotwin_size[1]],COLOR=2,/DEVICE
        plots, [0,state.plotwin_size[0]],[event.y,event.y],COLOR=2,/DEVICE
        
     end
     
  endcase

  if not state.freeze then begin
     
;     label = 'Cursor (X,Y): '+strtrim(xy[0],2)+', '+strtrim(xy[1],2)
;     widget_control,state.message,SET_VALUE=label
     
  endif
  
  widget_control, state.plotwin,/INPUT_FOCUS

  
end
;
;===============================================================================
;
pro xmergeorders_resizeevent,event

  common xmergeorders_state

  if n_params() eq 0 then begin
     
     size = widget_info(state.xmergeorders_base, /GEOMETRY)
     xsize = size.xsize
     ysize = size.ysize
     
  endif else begin
     
     widget_control, state.xmergeorders_base, TLB_GET_SIZE=size
     xsize = size[0]
     ysize = size[1]
     
  endelse

  state.plotwin_size[0] = xsize-state.winbuffer[0]
  state.plotwin_size[1] = ysize-state.winbuffer[1]
  
  widget_control, state.xmergeorders_base,UPDATE=0
  widget_control, state.plotwin, DRAW_XSIZE=state.plotwin_size[0], $
                  DRAW_YSIZE=state.plotwin_size[1]
  widget_control, state.xmergeorders_base,UPDATE=1
  
  

  wdelete,state.pixmap_wid
  window, /FREE, /PIXMAP,XSIZE=state.plotwin_size[0],YSIZE=state.plotwin_size[1]
  state.pixmap_wid = !d.window

  if state.freeze then begin

     erase, COLOR=20
     wset, state.plotwin_wid
     device, COPY=[0,0,state.plotwin_size[0],state.plotwin_size[1],0,0,$
                   state.pixmap_wid]
     
  endif else xmergeorders_plotspec

end  
;
;=============================================================================
;
;----------------------------- Main Program ---------------------------------
;
;=============================================================================
;
pro xmergeorders,filename,CANCEL=cancel

  if not xregistered('xmergeorders') then xmergeorders_initcommon

  common xmergeorders_state
  
  cleanplot,/SILENT
  
;  Get spextool and instrument information 
  
  mc_getspextoolinfo,spextoolpath,packagepath,spextool_keywords,instr,notspex, $
                     version,CANCEL=cancel
  if cancel then return
  
  mc_getosinfo,dirsep,strsep,CANCEL=cancel
  if cancel then return
  
;  Load color table
  
  mc_mkct
  device, RETAIN=2
  
;  Get fonts
  
  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return

;  Get screen size

  screensize = get_screen_size()
     
;  Build the structures

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
           buttonfont:buttonfont,$
           cenwave:ptr_new(2),$
           cliporder:-1,$
           charsize:1.5,$
           cursormode:'None',$
           flxxrange:[0.,0.],$
           flxyrange:[0.,0.],$
           flxyranges:ptr_new(2),$
           freeze:1,$
           instrument:instr.instrument,$
           ispectrum_fld:[0L,0L],$
           hdr:ptr_new(2),$
           mergedspec:ptr_new(2),$
           mbut_atmos:0L,$
           mbut_comb:0L,$
           mbut_flx:0L,$
           mbut_over:0L,$
           mbut_snr:0L,$
           mbut_unc:0L,$
           message:0L,$
           oname_fld:[0L,0L],$
           orders:ptr_new(2),$
           origspec:ptr_new(2),$
           naps:0L,$
           norders:0L,$           
           packagepath:packagepath,$
           path:'',$
           pixmap_wid:0L,$
           pixpp:250.0,$
           plotatmosphere:0,$
           plottype:'Overlapped',$
           plotwin_wid:0L,$
           plotwin:0L,$
           plotwin_size:[screensize[0]*0.5,screensize[1]*0.5],$
           pscale:!p,$
           reg:[[!values.d_nan,!values.d_nan],$
                [!values.d_nan,!values.d_nan]],$
           selectord_dl:0L,$
           slider:0L,$
           sliderval:50,$
           snlimit_fld:[0L,0L],$
           snlimit:ptr_new(2),$
           snrmask:ptr_new(2),$
           snrxrange:[0.,0.],$
           snryrange:[0.,0.],$
           snryranges:ptr_new(2),$
           spectype:'Flux',$
           spextoolpath:spextoolpath,$
           uncxrange:[0.,0.],$
           uncyrange:[0.,0.],$
           uncyranges:ptr_new(2),$
           usrmask:ptr_new(2),$
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
           workspec:ptr_new(2),$
           xmergeorders_base:0L}
  
  title = 'xmergeorders '+version+' for '+state.instrument
  
  state.xmergeorders_base = widget_base(TITLE=title, $
                                        /COLUMN,$
                                        /TLB_SIZE_EVENTS)
  
     quit_button = widget_button(state.xmergeorders_base,$
                                 FONT=buttonfont,$
                                 EVENT_PRO='xmergeorders_event',$
                                 VALUE='Quit',$
                                 UVALUE='Quit')
     
     row_base = widget_base(state.xmergeorders_base,$
                            /ROW)

        col1_base = widget_base(row_base,$
                                EVENT_PRO='xmergeorders_event',$
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
                                       EVENT_PRO='xmergeorders_event')
                 
               
                 input_fld = coyote_field2(row,$
                                           LABELFONT=buttonfont,$
                                           FIELDFONT=textfont,$
                                           TITLE=':',$
                                           UVALUE='Input Spectrum Field',$
                                           XSIZE=20,$
;                                           VALUE='spectra61-70.fits',$
                                           EVENT_PRO='xmergeorders_event',$
                                           /CR_ONLY,$
                                           TEXTID=textid)
                 state.ispectrum_fld = [input_fld,textid]

              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
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
                                   VALUE='3.  S/N Mask',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)

              input_fld = coyote_field2(box3_base,$
                                        LABELFONT=buttonfont,$
                                        FIELDFONT=textfont,$
                                        TITLE='S/N Limit:',$
                                        UVALUE='S/N Limit Field',$
                                        XSIZE=10,$
                                        EVENT_PRO='xmergeorders_event',$
                                        /CR_ONLY,$
                                        TEXTID=textid)
              state.snlimit_fld = [input_fld,textid]
              
           box4_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box4_base,$
                                   VALUE='4.  Select Order',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)

              state.selectord_dl = widget_droplist(box4_base,$
                                                   FONT=buttonfont,$
                                                   TITLE='Select Order: ',$
                                                   VALUE='01',$
                                                   UVALUE='Select Order Button')
              
           box5_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box5_base,$
                                   VALUE='5.  Write File',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)

              oname = coyote_field2(box5_base,$
                                    LABELFONT=buttonfont,$
                                    FIELDFONT=textfont,$
                                    TITLE='File Name:',$
                                    UVALUE='Object File Oname',$
                                    XSIZE=20,$
                                    TEXTID=textid)
            state.oname_fld = [oname,textid]
            
            write = widget_button(box5_base,$
                                  VALUE='Write File',$
                                  UVALUE='Write File',$
                                  FONT=buttonfont)
              

              
        col2_base = widget_base(row_base,$
                                FRAME=2,$
                                /COLUMN)

           row = widget_base(col2_base,$
                             /ROW,$
                             /BASE_ALIGN_CENTER,$
                             EVENT_PRO='xmergeorders_event')

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
                                                EVENT_PRO='xmergeorders_event',$
                                                UVALUE='Plot Flux Button',$
                                                FONT=buttonfont)
                 
                 state.mbut_unc = widget_button(subrow, $
                                                VALUE='Uncertainty', $
                                                /NO_RELEASE,$
                                                EVENT_PRO='xmergeorders_event',$
                                             UVALUE='Plot Uncertainty Button',$
                                                FONT=buttonfont)
                 
                 state.mbut_snr = widget_button(subrow, $
                                                VALUE='S/N', $
                                                /NO_RELEASE,$
                                                EVENT_PRO='xmergeorders_event',$
                                                UVALUE='Plot S/N Button',$
                                                FONT=buttonfont)
                 widget_control, state.mbut_flx,/SET_BUTTON

                 subrow = widget_base(row,$
                                      /ROW,$
                                      /TOOLBAR,$
                                      /NONEXCLUSIVE)

                 state.mbut_atmos = widget_button(subrow, $
                                                  VALUE='Atmosphere', $
                                                EVENT_PRO='xmergeorders_event',$
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
                                           EVENT_PRO='xmergeorders_event',$
                                           UVALUE='2 Color Button',$
                                           FONT=state.buttonfont)
                    if state.altcolor eq 2 then widget_control, button, $
                       /SET_BUTTON
              
                    button = widget_button(subrow, $
                                           VALUE='3 Color', $
                                           EVENT_PRO='xmergeorders_event',$
                                           UVALUE='3 Color Button',$
                                           FONT=state.buttonfont)
                    if state.altcolor eq 3 then widget_control, button, $
                       /SET_BUTTON
                 
              label = widget_label(row,$
                                   VALUE='Spectrum Type:',$
                                   FONT=buttonfont)

                 subrow = widget_base(row,$
                                      /ROW,$
                                      /TOOLBAR,$
                                      /EXCLUSIVE)
                 
                 state.mbut_over = widget_button(subrow, $
                                                 VALUE='Overlapping', $
                                                /NO_RELEASE,$
                                                EVENT_PRO='xmergeorders_event',$
                                               UVALUE='Plot Overlapped Button',$
                                                 FONT=buttonfont)
                 widget_control, state.mbut_over,/SET_BUTTON
                 
                 state.mbut_comb = widget_button(subrow, $
                                                /NO_RELEASE,$
                                                 VALUE='Merged', $
                                                EVENT_PRO='xmergeorders_event',$
                                                 UVALUE='Plot Merged Button',$
                                                 FONT=buttonfont)

              label = widget_label(row,$
                                   VALUE='  ')
                                  
           row = widget_base(col2_base,$
                             /ROW,$
                             /BASE_ALIGN_CENTER)
           
              state.plotwin = widget_draw(row,$
                                          /ALIGN_CENTER,$
                                          XSIZE=state.plotwin_size[0],$
                                          YSIZE=state.plotwin_size[1],$
                                       EVENT_PRO='xmergeorders_plotwinevent',$
                                          /KEYBOARD_EVENTS,$
                                          /BUTTON_EVENTS,$
                                          /TRACKING_EVENTS,$
                                          /MOTION_EVENTS)
              
           state.slider = widget_slider(col2_base,$
                                        UVALUE='Slider',$
                                        EVENT_PRO='xmergeorders_event',$
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
                                EVENT_PRO='xmergeorders_minmaxevent',$
                                /CR_ONLY,$
                                TEXTID=textid)
           state.xmin_fld = [xmin,textid]
           
           xmax = coyote_field2(row_base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='X Max:',$
                                UVALUE='X Max',$
                                XSIZE=15,$
                                EVENT_PRO='xmergeorders_minmaxevent',$
                                /CR_ONLY,$
                                TEXTID=textid)
           state.xmax_fld = [xmax,textid]
           
           ymin = coyote_field2(row_base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Y Min:',$
                                UVALUE='Y Min',$
                                XSIZE=15,$
                                EVENT_PRO='xmergeorders_minmaxevent',$
                                /CR_ONLY,$
                                TEXTID=textid)
           state.ymin_fld = [ymin,textid]
           
           ymax = coyote_field2(row_base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Y Max:',$
                                UVALUE='Y Max',$
                                XSIZE=15,$
                                EVENT_PRO='xmergeorders_minmaxevent',$
                                /CR_ONLY,$
                                TEXTID=textid)
           state.ymax_fld = [ymax,textid]
           
   button = widget_button(state.xmergeorders_base,$
                          FONT=buttonfont,$
                          EVENT_PRO='xmergeorders_event',$
                          VALUE='Help',$
                          UVALUE='Help')

           
; Get things running.  Center the widget using the Fanning routine.
      
    cgcentertlb,state.xmergeorders_base
      
    widget_control, state.xmergeorders_base, /REALIZE

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
   
   widget_geom = widget_info(state.xmergeorders_base, /GEOMETRY)
   
   state.winbuffer[0]=widget_geom.xsize-state.plotwin_size[0]
   state.winbuffer[1]=widget_geom.ysize-state.plotwin_size[1]
    
; Start the Event Loop. 
    
   XManager, 'xmergeorders', $
             state.xmergeorders_base, $
             CLEANUP='xmergeorders_cleanup',$
             EVENT_HANDLER='xmergeorders_resizeevent',$
             /NO_BLOCK
   
   if n_params() ne 0 then begin

      widget_control, state.ispectrum_fld[1],SET_VALUE=filename
      tic
      xmergeorders_loadspec
      toc
      
   endif

   
cont:

end
