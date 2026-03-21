;+
; NAME:
;     xmc_findshifts
;
; PURPOSE:
;     To determine pixel shifts between object and telluric spectra
;
; CALLING SEQUENCE:
;     result = xmc_findshifts(objspec,telspec,orders,awave,atrans,xtitle, $
;                             CANCEL=cancel)
;
; INPUTS:
;     objspec - A [nwave,4,norders*naps] array of object spectra.
;     telspec - A [nwave,4,norders*naps] array of telluric correction
;               spectra.
;     orders  - An array of order numbers.
;     awave   - Wavelengths for the atmospheric transmission
;               (microns).
;     atrans  - Transmission for the atmospheric transmission
;     xitlte  - A string giving the xtitle 
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return if there is a problem.
;
; OUTPUTS:
;     result - A [norders,naps] array giving the shifts (in pixels)
;              for each order.
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     findshifts_state     
;
; RESTRICTIONS:
;     None
;
; DEPENDENCIES:
;     Spextool library (and its dependencies)
;
; PROCEDURE:
;     GUI
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-10-08 - Written by M. Cushing, University of Toledo
;-
;
;===============================================================================
;
; ----------------------------Support procedures------------------------------ 
;
;===============================================================================
;
pro findshifts_initcommon,objspec,telspec,orders,awave,atrans,xtitle, $
                          CANCEL=cancel

  norders = n_elements(orders)
  naps = n_elements(objspec[0,0,*])/norders
  screensize = get_screen_size()
  
  wranges = fltarr(2,norders,/NOZERO)
  yranges = fltarr(2,norders,naps,/NOZERO)
  cenwave = fltarr(norders,/NOZERO)
  
;  Get the data ready

  l = 0
  for i = 0,norders-1 do begin

     for j = 0,naps-1 do begin

        owave = objspec[*,0,i*naps+j]
        oflux = objspec[*,1,i*naps+j]
        trim = mc_nantrim(owave,2)  
        owave = owave[trim]
        oflux = oflux[trim]
        
        twave = telspec[*,0,i]
        tflux = telspec[*,1,i]
        trim = mc_nantrim(twave,2)  
        twave = twave[trim]
        tflux = tflux[trim]
        
        linterp,twave,tflux,owave,ntflux,MISSING=!values.f_nan
        div = oflux*ntflux
        
;  Get plot ranges, smooth to avoid bad pixels
     
        smooth = mc_robustsg(findgen(n_elements(owave)),div,5,3,0.1, $
                             CANCEL=cancel)
        if cancel then return
        
        min = min(smooth[*,1],/NAN,MAX=max)
        yranges[*,i,j] = mc_bufrange([min,max],0.1)
        
;  Now do the wavelengths 
        
        wranges[*,i] = [min(owave,MAX=max,/NAN),max]
        cenwave[i] = total(wranges[*,i])/2.
        
;  Store the data
        
        arr = [[owave],[oflux],[ntflux],[div],[div]]
        tag = 'O'+string(i+1,FORMAT='(I3.3)')+'AP'+string(j+1,FORMAT='(I2.2)')
        data = (l eq 0) ? create_struct(tag,arr):create_struct(data,tag,arr)
        l++
        
     endfor

  endfor
     

  wrange = [min(wranges,MAX=max),max]
  yrange = [min(yranges,MAX=max),max]
    
  cleanplot,/SILENT

  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return
    
  common findshifts_state, state

  state = {abswrange:wrange,$
           absyrange:yrange,$
           allrow:0L,$
           altcolor:2,$
           ap:0,$
           atmosthresh:0.0,$
           atrans:atrans,$
           awave:awave,$
           buttonfont:buttonfont,$
           cancel:0L,$
           cenwave:cenwave,$
           charsize:1.5,$
           corwranges:make_array(2,norders,naps,/FLOAT,VALUE=!values.f_nan),$
           cursormode:'None',$
           data:data,$
           findshifts_base:0L,$
           focusorder:-1,$
           mode:'All',$
           menu:0L,$
           naps:naps,$
           norders:norders,$
           orderrow:0L,$
           orders:orders,$
           orderstring:'',$
           pixmap1_wid:0L,$
           pixmap2_wid:0L,$
           plotatmosphere:0,$
           plotshifts:0,$
           plotshiftedspectra:1,$
           plotwin1:0L,$
           plotwin1_wid:0L,$
           plotwin2:0L,$
           plotwin2_wid:0L,$
           plotsize:screensize*[0.6,0.3],$
           pltshifts_but:0L,$
           pltspecs_but:0L,$
           pscale:!p,$
           reg:[[!values.d_nan,!values.d_nan],$
                [!values.d_nan,!values.d_nan]],$
           scroll_size:screensize*[0.5,0.5],$
           selectap_dl:0L,$
           selectordr_dl:0L,$
           shift_fld:[0L,0L],$
           shifts:replicate(!values.f_nan,norders,naps),$
           slider:0L,$
           sliderval:50,$
           textfont:textfont,$
           orders_fld:[0L,0L],$
           winbuffer:[0L,0L],$
           wrange:wrange,$
           wranges:wranges,$
           xmax_fld:[0L,0L],$
           xmin_fld:[0L,0L],$
           xscale:!x,$
           xtitle:xtitle,$
           ymax_fld:[0L,0L],$
           ymin_fld:[0L,0L],$
           yrange:yrange,$
           yranges:yranges,$
           yscale:!y}

end
;
;===============================================================================
;
pro findshifts_findshifts,zordr

  common findshifts_state

  new = 0
  zdata = zordr*state.naps+state.ap
  
  norders = n_elements(zordr)
  for i = 0,norders-1 do begin

;  If the order hasn't been done, get set up     
     
     if finite(state.corwranges[0,zordr[i],state.ap]) eq 0 then begin

        state.corwranges[*,zordr[i],state.ap] = $
           mc_bufrange(state.wranges[*,zordr[i]],-0.1)
        
     endif

;  Get the atmosphere

     linterp,state.awave,state.atrans,state.data.(zdata[i])[*,0],atmosphere, $
             MISSING=0

; Figure out the pixels in the wavelength range
     
     zwave = where(state.data.(zdata[i])[*,0] gt $
                   state.corwranges[0,zordr[i],state.ap] and $
                   state.data.(zdata[i])[*,0] lt $
                   state.corwranges[1,zordr[i],state.ap],cnt)
          
     x = findgen(n_elements(state.data.(zdata[i])[*,0]))
     subx = findgen(cnt)
     
     if new then begin
        
        oversamp = 1
        lagwin = 3
         
        xx = findgen(cnt*oversamp)/float(oversamp)
        linterp,subx,state.data.(zdata[i])[*,1],xx,obj
        linterp,subx,state.data.(zdata[i])[*,2],xx,tel
        
        lag = findgen(2*lagwin*oversamp+1)/oversamp-lagwin
        corr = c_correlate(state.data.(zdata[i])[*,1],1/ $
                           state.data.(zdata[i])[*,2],lag)
        
        coeff = mc_polyfit1d(lag,corr,2,/SILENT)
        
        shift = -1*(-coeff[1]/2./coeff[2])

     endif else begin

        del = 10

        shifts = findgen(151)/50.+(0-1.5)
        rms    = fltarr(151)
        
        if norders eq 1 then begin
        
           wset, state.pixmap2_wid
           plot,[1],[1],/XSTYLE,XTITLE=state.xtitle,TITLE='Shifted Spectra', $
                YTITLE='Relative Intensity',CHARSIZE=state.charsize,/NODATA,$
                YRANGE=state.yrange,XRANGE=state.wrange,/YSTYLE, $
                BACKGROUND=20,NOERASE=noerase, $
                POSITION=[120,60,state.plotsize[0]-40,state.plotsize[1]-50], $
                /DEVICE

        endif
        
        for j = 0, 150 do begin

           if norders eq 1 then widget_control, $
              state.shift_fld[1], SET_VALUE=string(shifts[j],FORMAT='(f+5.2)')
           
           linterp,subx+shifts[j],state.data.(zdata[i])[zwave,2],subx,shifted, $
                   MISSING=!values.f_nan
           div = state.data.(zdata[i])[zwave,1]*shifted
           med = median(div,/EVEN)
           mad = 1.482*median(abs(div-med), /EVEN)
           rms[j] = mad
           
           if norders eq 1 then begin
           
              wset, state.plotwin2_wid
              device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0, $
                            state.pixmap2_wid]

              case state.altcolor of
                 
                 2: color = (zordr[i] mod 2) ? 1:3
                 
                 3: begin
                    
                    case zordr[i] mod 3 of
                       
                       0: color=3
                       
                       1: color=1
                       
                       2: color=2
                       
                    endcase
                    
                 end
                 
              endcase
                            
              linterp,x+shifts[j],state.data.(zdata[i])[*,2],x,tmp, $
                      MISSING=!values.f_nan
              div = state.data.(zdata[i])[*,1]*tmp
              
              oplot,state.data.(zdata[i])[*,0],div,COLOR=color,PSYM=10
              wait, 0.00105

           endif
          
        endfor

        min = min(rms,minidx)

        coeff = mc_polyfit1d(shifts[(minidx-del) >0:(minidx+del) < 149],$
                             rms[(minidx-del) > 0:(minidx+del) < 149],2, $
                             /SILENT)

; Debug plotting code
;
;        window, 2
;        plot,shifts,rms,PSYM=-1,/XSTY,/YSTY
;        oplot,shifts[(minidx-del) >0:(minidx+del) < 149], $
;              poly(shifts[(minidx-del) >0:(minidx+del) < 149],coeff), $
;              COLOR=2
        
        shift = -coeff[1]/2./coeff[2]
        if shift lt -1.5 then shift = -1.5
        if shift gt 1.5 then shift = 1.5

        
     endelse
        
     state.shifts[zordr[i],state.ap] = shift
     if norders eq 1 then widget_control, $
        state.shift_fld[1], SET_VALUE=string(shift,FORMAT='(f+5.2)')

  endfor
  
end
;
;===============================================================================
;
pro findshifts_mkmenu,DESTROY=destroy

  common findshifts_state

  if state.mode eq 'All' then begin

     if keyword_set(DESTROY) then begin

        widget_control, state.menu, UPDATE=0
        widget_control, state.orderrow,/DESTROY

     endif

     widget_control, state.pltshifts_but,SENSITIVE=1
     widget_control, state.pltspecs_but,SENSITIVE=1
     
     state.allrow = widget_base(state.menu,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
     
        fld = coyote_field2(state.allrow,$
                            EVENT_PRO='findshifts_event',$
                            /CR_ONLY,$
                            LABELFONT=state.buttonfont,$
                            FIELDFONT=state.textfont,$
                            TITLE='Orders:',$
                            UVALUE='Orders Field',$
                            XSIZE=20,$
                            VALUE=state.orderstring,$
                            TEXTID=textid) 
        state.orders_fld = [fld,textid]

        button = widget_button(state.allrow,$
                               FONT=state.buttonfont,$
                               VALUE='Clear Shifts',$
                               UVALUE='Clear Shifts')

        values = ['None',strtrim(string(state.orders,FORMAT='(I3)'),2)]
        state.selectordr_dl = widget_droplist(state.allrow,$
                                              FONT=state.buttonfont,$
                                              TITLE='Select Order: ',$
                                              VALUE=values,$
                                              UVALUE='Select Order Button')
     
  endif else begin

     if keyword_set(DESTROY) then begin

        widget_control, state.menu, UPDATE=0
        widget_control, state.allrow,/DESTROY

     endif

     state.plotshifts = 0
     state.plotshiftedspectra = 1
     widget_control, state.pltshifts_but,SET_BUTTON=0,SENSITIVE=0
     widget_control, state.pltspecs_but,/SET_BUTTON,SENSITIVE=0
          
     state.orderrow = widget_base(state.menu,$
                                  /ROW,$
                                  /BASE_ALIGN_CENTER)
     
        fld = coyote_field2(state.orderrow,$
                            EVENT_PRO='findshifts_event',$
                            /CR_ONLY,$
                            LABELFONT=state.buttonfont,$
                            FIELDFONT=state.textfont,$
                            TITLE='Shift (pixels):',$
                            UVALUE='Shift (pixels) Field',$
                            XSIZE=5,$
                            TEXTID=textid) 
        state.shift_fld = [fld,textid]

        button = widget_button(state.orderrow,$
                               FONT=state.buttonfont,$
                               VALUE=' + ',$
                               UVALUE='+ Button')
        
        button = widget_button(state.orderrow,$
                               FONT=state.buttonfont,$
                               VALUE=' - ',$
                               UVALUE='- Button')

        button = widget_button(state.orderrow,$
                               FONT=state.buttonfont,$
                               VALUE='Apply to All Orders/Apertures',$
                               UVALUE='Apply to All Orders Button')
        
        button = widget_button(state.orderrow,$
                               FONT=state.buttonfont,$
                               VALUE='Done',$
                               UVALUE='Done Button')
                            
  endelse

  if keyword_set(DESTROY) then widget_control, state.menu, UPDATE=1
  
end
;
;===============================================================================
;
pro findshifts_plotspec1

  common findshifts_state

  wset, state.pixmap1_wid
  position = [120,60,state.plotsize[0]-40,state.plotsize[1]-50]

  noerase = 0
  ystyle = 1
  
  if state.plotatmosphere then begin

     plot,[1],[1],XSTYLE=5,/NODATA,YRANGE=[0,1],YSTYLE=5,$
          CHARSIZE=state.charsize,XRANGE=state.wrange,$
          BACKGROUND=20,POSITION=position,/DEVICE             
     z = where(state.awave ge state.wrange[0] and $
               state.awave le state.wrange[1])
                   
     oplot,state.awave[z],state.atrans[z],COLOR=5,PSYM=10
     ystyle = 9
     noerase = 1
     
  endif 

  plot,[1],[1],/XSTYLE,XTITLE=state.xtitle,YTITLE='Relative Intensity',$
       CHARSIZE=state.charsize,/NODATA,$
       YRANGE=state.yrange,XRANGE=state.wrange,YSTYLE=ystyle, $
       BACKGROUND=20,NOERASE=noerase,POSITION=position,/DEVICE

  xyouts,20,state.plotsize[1]-30,'!5Order',ALIGNMENT=0,/DEVICE, $
         CHARSIZE=state.charsize
    
  for i = 0,state.norders-1 do begin

     idx = i*state.naps+state.ap
     
     if ~mc_rangeinrange(state.wranges[*,i],!x.crange) then continue
     if state.focusorder ne -1 then if i ne state.focusorder then continue

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
     
     oplot,state.data.(idx)[*,0],state.data.(idx)[*,3],COLOR=color,PSYM=10

;  Plot the lines 
     
     if state.focusorder ne -1 then begin

        plots,[state.corwranges[0,i,state.ap],state.corwranges[0,i,state.ap]], $
              !y.crange,COLOR=7,LINESTYLE=2
        plots,[state.corwranges[1,i,state.ap],state.corwranges[1,i,state.ap]], $
              !y.crange,COLOR=7,LINESTYLE=2

     endif
     
;  Label order numbers
        
     min = max([!x.crange[0],(state.wranges)[0,i]])
     max = min([!x.crange[1],(state.wranges)[1,i]])
     
     lwave = (min+max)/2.
     
     if lwave gt !x.crange[0] then begin
        
        xy = convert_coord(lwave,!y.crange[1],/DATA,/TO_DEVICE)
        
        xyouts,xy[0],xy[1]+10+15*(i mod 2), $
               strtrim(string((state.orders)[i],FORMAT='(I3)'),2), $
               /DEVICE,COLOR=color,ALIGNMENT=0.5,CHARSIZE=state.charsize
        
     endif

     if !y.crange[0] lt 0 then plots,!x.crange,[0,0],LINESTYLE=1
     
  endfor

  if ystyle eq 9 then begin

     ticks = ['0.0','0.2','0.4','0.6','0.8','1.0']
     axis,YAXIS=1,YTICKS=5,YTICKNAME=ticks,YMINOR=2,COLOR=5, $
          CHARSIZE=state.charsize

  endif

  state.xscale = !x
  state.yscale = !y
  state.pscale = !p
  
end
;
;===============================================================================
;
pro findshifts_plotspec2

  common findshifts_state

  wset, state.pixmap2_wid
  position = [120,60,state.plotsize[0]-40,state.plotsize[1]-50]

  noerase = 0
  ystyle = 1

  if state.plotshiftedspectra then begin

     plot,[1],[1],/XSTYLE,XTITLE=state.xtitle,YTITLE='Relative Intensity',$
          CHARSIZE=state.charsize,/NODATA,$
          YRANGE=state.yrange,XRANGE=state.wrange,YSTYLE=ystyle, $
          BACKGROUND=20,NOERASE=noerase,POSITION=position,/DEVICE, $
          TITLE='Shifted Spectra'

  endif else begin
     
     plot,[1],[1],/XSTYLE,XTITLE=state.xtitle,YTITLE='Shifts (pixels)',$
          CHARSIZE=state.charsize,/NODATA,$
          YRANGE=[-1.5,1.5],XRANGE=state.wrange,YSTYLE=ystyle, $
          BACKGROUND=20,NOERASE=noerase,POSITION=position,/DEVICE, $
          TITLE='Shifts'
     
  endelse
  
  for i = 0,state.norders-1 do begin

     idx = i*state.naps+state.ap
     
     if ~mc_rangeinrange(state.wranges[*,i],!x.crange) then continue
     if state.focusorder ne -1 then if i ne state.focusorder then continue

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
     
     if state.plotshiftedspectra then begin

        oplot,state.data.(idx)[*,0],state.data.(idx)[*,4],COLOR=color,PSYM=10

     endif else begin

        plotsym,0,1,/FILL
        if state.cenwave[i] gt state.wrange[0] and $
           state.cenwave[i] lt state.wrange[1] then $
              plots,state.cenwave[i],state.shifts[i,state.ap],COLOR=color, $
                    PSYM=8
        
     endelse
     if !y.crange[0] lt 0 then plots,!x.crange,[0,0],LINESTYLE=1

;  Plot the lines 
     
     if state.focusorder ne -1 then begin

        plots,[state.corwranges[0,i,state.ap],state.corwranges[0,i,state.ap]], $
              !y.crange,COLOR=7,LINESTYLE=2
        plots,[state.corwranges[1,i,state.ap],state.corwranges[1,i,state.ap]], $
              !y.crange,COLOR=7,LINESTYLE=2

     endif
     
  endfor
  
end
;
;===============================================================================
;
pro findshifts_plotupdate,MINMAX=minmax

  common findshifts_state

  mc_mkct
  findshifts_plotspec1
  wset, state.plotwin1_wid
  device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0,state.pixmap1_wid]
  
  findshifts_plotspec2
  wset, state.plotwin2_wid
  device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0,state.pixmap2_wid]
  
  if keyword_set(MINMAX) then findshifts_setminmax

  
end
;
;===============================================================================
;
pro findshifts_shiftspectra,zordr

  common findshifts_state

  zdata = zordr*state.naps+state.ap

  ;03/12/2020 A.Boogert: corrected type zrdr should be zordr
  ;norders = n_elements(zrdr)
  norders = n_elements(zordr)
  for i = 0,norders-1 do begin
     
     x = findgen(n_elements(state.data.(zordr[i])[*,0]))

     linterp,x+state.shifts[zdata[i]],state.data.(zdata[i])[*,2],x,new
     state.data.(zdata[i])[*,4] = state.data.(zdata[i])[*,1]*new

  endfor

end
;
;===============================================================================
;
pro findshifts_selectorder

  common findshifts_state
  
  state.wrange = mc_bufrange(state.wranges[*,state.focusorder],0.05)
  state.yrange = mc_bufrange(state.yranges[*,state.focusorder,state.ap],0.05)
  
;  Check whether this order has been shifted already
  
  if finite(state.shifts[state.focusorder,state.ap]) then begin
     
     widget_control, state.shift_fld[1], $
                     SET_VALUE=string(state.shifts[state.focusorder,state.ap], $
                                      FORMAT='(f+5.2)')
     
  endif
  
  findshifts_plotupdate
  state.cursormode = 'None'
   
end
;
;===============================================================================
;
pro findshifts_setminmax

  common findshifts_state
  
  widget_control, state.xmin_fld[1],SET_VALUE=strtrim(state.wrange[0],2)
  
  widget_control, state.xmax_fld[1],SET_VALUE=strtrim(state.wrange[1],2)
  
  widget_control, state.ymin_fld[1],SET_VALUE=strtrim(state.yrange[0],2)
  
  widget_control, state.ymax_fld[1],SET_VALUE=strtrim(state.yrange[1],2)
  
  findshifts_setslider
  
end
;
;=============================================================================
;
pro findshifts_setslider

  common findshifts_state

  del = state.abswrange[1]-state.abswrange[0]
  midwave = (state.wrange[1]+state.wrange[0])/2.
  state.sliderval = round((midwave-state.abswrange[0])/del*100)
  
  widget_control, state.slider, SET_VALUE=state.sliderval
  
end
;
;=============================================================================
;
pro findshifts_zoom,IN=in,OUT=out

  common findshifts_state

  delabsx = state.abswrange[1]-state.abswrange[0]
  delx    = state.wrange[1]-state.wrange[0]
  
  delabsy = state.absyrange[1]-state.absyrange[0]
  dely    = state.yrange[1]-state.yrange[0]
  
  xcen = state.wrange[0]+delx/2.
  ycen = state.yrange[0]+dely/2.
  
  case state.cursormode of 
     
     'XZoom': begin
        
        z = alog10(delabsx/delx)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsx/2.^z/2.
        state.wrange = [xcen-hwin,xcen+hwin]
        
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
        state.wrange = [xcen-hwin,xcen+hwin]
        
        z = alog10(delabsy/dely)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsy/2.^z/2.
        state.yrange = [ycen-hwin,ycen+hwin]
        
     end
     
     else:
     
endcase

  findshifts_plotupdate,/MINMAX
  
end
;
;===============================================================================
;
; ------------------------------Event Handlers-------------------------------- 
;
;===============================================================================
;
pro findshifts_event, event

  common findshifts_state

  widget_control, event.id,  GET_UVALUE=uvalue
  
  case uvalue of

     '2 Color Button': begin

        state.altcolor = 2
        findshifts_plotupdate
        
     end

     '3 Color Button': begin

        state.altcolor = 3
        findshifts_plotupdate
        
     end
     
     '+ Button': begin

        offset = mc_cfld(state.shift_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        offset = offset+0.02

        offset = mc_crange(offset,[-1.5,1.5],'Shift',/KGE,/KLE, $
                           WIDGET_ID=state.findshifts_base,CANCEL=cancel)
        if cancel then begin

           widget_control, state.shift_fld[1], $
              SET_VALUE=string(state.shifts[state.focusorder],FORMAT='(f+5.2)')

        endif else begin
           
           widget_control,state.shift_fld[1], $
              SET_VALUE=string(offset,FORMAT='(f+5.2)')
           state.corwranges[*,state.focusorder] = !values.f_nan
           state.shifts[state.focusorder] = offset
           findshifts_shiftspectra,state.focusorder
           findshifts_plotupdate

        endelse
           
     end

     '- Button': begin

        offset = mc_cfld(state.shift_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        offset = offset-0.02

        offset = mc_crange(offset,[-1.5,1.5],'Shift',/KGE,/KLE, $
                           WIDGET_ID=state.findshifts_base,CANCEL=cancel)
        if cancel then begin

           widget_control, state.shift_fld[1], $
              SET_VALUE=string(state.shifts[state.focusorder],FORMAT='(f+5.2)')

        endif else begin
           
           widget_control,state.shift_fld[1], $
              SET_VALUE=string(offset,FORMAT='(f+5.2)')
           state.corwranges[*,state.focusorder] = !values.f_nan
           state.shifts[state.focusorder] = offset
           findshifts_shiftspectra,state.focusorder
           findshifts_plotupdate

        endelse
           
     end
     
     'Accept': widget_control, event.top, /DESTROY

     'Apply to All Orders Button': begin

        offset = mc_cfld(state.shift_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        
        offset = mc_crange(offset,[-1.5,1.5],'Shift',/KGE,/KLE, $
                           WIDGET_ID=state.findshifts_base,CANCEL=cancel)
        if cancel then return

        state.shifts = offset
        state.focusorder = -1
        state.mode = 'All'
        findshifts_mkmenu,/DESTROY
        findshifts_shiftspectra,indgen(n_elements(state.norders))        
        findshifts_plotupdate
        
     end
     
     'Orders Field': begin

        orders = mc_cfld(state.orders_fld,7,/EMPTY,CANCEL=cancel)
        if cancel then return

        state.orderstring = orders

        orders = mc_fsextract(orders,/INDEX,CANCEL=cancel)
        if cancel then return
        orders = orders[sort(orders)]

        match,state.orders,orders,idx,COUNT=cnt
        if cnt ne 0 then begin
                
           widget_control, /HOURGLASS        
           state.shifts = !values.f_nan
           findshifts_findshifts,idx
           findshifts_shiftspectra,idx
           findshifts_plotupdate        

        endif else begin

           ok = dialog_message('No orders found.',/INFORMATION,$
                               DIALOG_PARENT=state.findshifts_base)
           mc_setfocus,state.orders_fld
           
        endelse
           
     end
     
     'Cancel': begin

        state.cancel = 1
        widget_control, event.top, /DESTROY

     end

     'Clear Shifts': begin

        result = dialog_message('Are you sure you want to clear the shifts?',$
                                /QUESTION,DIALOG_PARENT=state.findshifts_base)
        if result ne 'No' then begin
           
           state.shifts = !values.f_nan
           widget_control, state.orders_fld[1],SET_VALUE=''
           for i = 0,state.norders-1 do $
              state.data.(i)[*,4] = state.data.(i)[*,3]
           findshifts_plotupdate        

     endif
        
     end
          
     'Done Button': begin

        state.focusorder = -1
        state.mode = 'All'
        findshifts_mkmenu,/DESTROY
        findshifts_plotupdate
        
     end

     'Find All Shifts': begin

     end
     
     'Plot Atmosphere Button': begin

        state.plotatmosphere = ~state.plotatmosphere
        findshifts_plotupdate

     end

     'Plot Shifted Spectra Button': begin

        state.plotshiftedspectra = ~state.plotshiftedspectra
        findshifts_plotupdate
        
     end
     
     'Plot Shifts Button': begin

        state.plotshifts = ~state.plotshifts
        findshifts_plotupdate

     end

     'Select Ap Button': begin
        
        state.ap = event.index
        findshifts_plotupdate

     end
     
     'Select Order Button': begin        
        
        state.focusorder = event.index-1
        if state.focusorder eq -1 then return
        state.mode = 'Order'
        findshifts_mkmenu,/DESTROY
        findshifts_selectorder

     end
     
     'Shift (pixels) Field': begin

        offset = mc_cfld(state.shift_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return

        offset = mc_crange(offset,[-9,9],'Shift',/KGE,/KLE, $
                           WIDGET_ID=state.findshifts_base,CANCEL=cancel)
        if cancel then begin

           widget_control, $
              state.shift_fld[1], $
              SET_VALUE=string(state.shifts[state.focusorder],FORMAT='(f+5.2)')

        endif else begin

           state.corwranges[*,state.focusorder] = !values.f_nan
           state.shifts[state.focusorder] = offset
           findshifts_shiftspectra,state.focusorder
           findshifts_plotupdate

        endelse
           
     end

     else:

  endcase

end
;
;===============================================================================
;
pro findshifts_minmaxevent,event

  common findshifts_state
  
  widget_control, event.id,  GET_UVALUE = uvalue
  
  case uvalue of 
     
     'X Min': begin
        
        xmin = mc_cfld(state.xmin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        xmin2 = mc_crange(xmin,state.wrange[1],'X Min',/KLT,$
                          WIDGET_ID=state.findshifts_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control, state.xmin_fld[0],SET_VALUE=state.wrange[0]
           return
           
        endif else state.wrange[0] = xmin2
        
     end

     'X Max': begin
        
        xmax = mc_cfld(state.xmax_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        xmax2 = mc_crange(xmax,state.wrange[0],'X Max',/KGT,$
                       WIDGET_ID=state.findshifts_base,CANCEL=cancel)
        if cancel then begin
            
            widget_control, state.xmax_fld[0],SET_VALUE=state.wrange[1]
            return
            
         endif else state.wrange[1] = xmax2
        
     end
     
     'Y Min': begin
        
        ymin = mc_cfld(state.ymin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        ymin2 = mc_crange(ymin,state.yrange[1],'Y Min',/KLT,$
                          WIDGET_ID=state.findshifts_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control,state.ymin_fld[0],SET_VALUE=state.yrange[0]
            return
            
        endif else state.yrange[0] = ymin2
        
     end
     
     'Y Max': begin

        ymax = mc_cfld(state.ymax_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        ymax2 = mc_crange(ymax,state.yrange[0],'Y Max',/KGT,$
                          WIDGET_ID=state.findshifts_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control,state.ymax_fld[0],SET_VALUE=state.yrange[1]
           return
           
        endif else state.yrange[1] = ymax2
        
     end
     
  endcase
  
  findshifts_plotupdate,/MINMAX
  
end

;
;===============================================================================
;
pro findshifts_plotwinevent1,event

  common findshifts_state

  widget_control, event.id,  GET_UVALUE=uvalue
  
  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin

     widget_control, state.plotwin1,INPUT_FOCUS=event.enter
     
     wset, state.plotwin1_wid
     device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0, $
                   state.pixmap1_wid]
     wset, state.plotwin2_wid
     device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0, $
                   state.pixmap2_wid]
     return
     
  endif

  ;  Check for arrow keys

  if event.type eq 6 and event.release eq 0 then begin

     case event.key of
              
        5: begin

           del = (state.wrange[1]-state.wrange[0])*0.3
           oldcen = (state.wrange[1]+state.wrange[0])/2.
           newcen = oldcen-del

           if newcen lt state.abswrange[0] then return
           state.wrange = state.wrange + (newcen-oldcen)
           findshifts_plotupdate,/MINMAX           
           
        end
        
        6: begin

           del = (state.wrange[1]-state.wrange[0])*0.3
           oldcen = (state.wrange[1]+state.wrange[0])/2.
           newcen = oldcen+del

           if newcen gt state.abswrange[1] then return
           state.wrange = state.wrange + (newcen-oldcen)
           findshifts_plotupdate,/MINMAX
           
        end
        
        else:
        
     endcase
     
  endif
     
  !p = state.pscale
  !x = state.xscale
  !y = state.yscale
  
;
;  Check for ASCII keyboard event
;
  if event.type eq 5 and event.release eq 1 then begin
  
     case strtrim(event.ch,2) of 

        'a': begin
           
           state.abswrange = state.wrange
           state.absyrange=state.yrange
           
        end
        
        'c': begin          
           
           state.cursormode = 'None'
           state.reg = !values.f_nan                
           findshifts_plotupdate
           
        end

        'd': begin

           state.focusorder = -1
           state.mode = 'All'
           findshifts_mkmenu,/DESTROY
           findshifts_plotupdate

        end
        
        'i': findshifts_zoom,/IN
        
        'o': findshifts_zoom,/OUT

        's': begin

           if state.focusorder eq -1 then begin

;  Find which order you are talking about              
              
              xydev = convert_coord(state.cenwave,replicate(1.0,state.norders),$
                                    /DATA,/TO_DEVICE)
              min = min(abs(reform(xydev[0,*])-event.x),z)
              state.focusorder = z              
              state.mode = 'Order'
              findshifts_mkmenu,/DESTROY
              findshifts_selectorder
              
           endif else begin
              
              state.corwranges[*,state.focusorder] = !values.f_nan
              findshifts_plotupdate,/MINMAX              

              state.cursormode = 'Select'
              state.reg = !values.f_nan
              
           endelse
                         
        end
        
        'w': begin
           
           state.wrange = state.abswrange
           state.yrange = state.absyrange
           findshifts_plotupdate,/MINMAX
           
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
     return
     
  endif 

  wset, state.plotwin1_wid
  x  = event.x/float(state.plotsize[0])
  y  = event.y/float(state.plotsize[1])
  xy = convert_coord(x,y,/NORMAL,/TO_DATA,/DOUBLE)

  if event.type eq 1 then begin

     z = where(finite(state.reg) eq 1,count)
     if count eq 0 then begin
        
        wset, state.pixmap1_wid
        state.reg[*,0] = xy[0:1]
        case state.cursormode of
           
           'XZoom': begin

              plots, [event.x,event.x],[0,state.plotsize[1]], $
                     COLOR=2,/DEVICE,LINESTYLE=2

              wset, state.pixmap2_wid
              
              plots, [event.x,event.x],[0,state.plotsize[1]], $
                     COLOR=2,/DEVICE,LINESTYLE=2
              
           end
           
           'YZoom': plots, [0,state.plotsize[0]],[event.y,event.y], $
                           COLOR=2,/DEVICE,LINESTYLE=2

           'Select': begin

              plots, [event.x,event.x],[0,state.plotsize[1]], $
                     COLOR=7,/DEVICE,LINESTYLE=2
              
              wset, state.pixmap2_wid
              
              plots, [event.x,event.x],[0,state.plotsize[1]], $
                     COLOR=7,/DEVICE,LINESTYLE=2

           end
              
           else:
           
        endcase
        wset, state.plotwin1_wid
        device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0, $
                      state.pixmap1_wid]

        wset, state.plotwin2_wid
        device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0, $
                      state.pixmap2_wid]
        
     endif else begin 
        
        state.reg[*,1] = xy[0:1]
        case state.cursormode of 
           
           'XZoom': state.wrange = [min(state.reg[0,*],MAX=max),max]
           
           'YZoom': state.yrange = [min(state.reg[1,*],MAX=max),max]
           
           'Zoom': begin
              
              state.wrange = [min(state.reg[0,*],MAX=max),max]
              state.yrange = [min(state.reg[1,*],MAX=max),max]
              
           end

           'Select': begin

              state.corwranges[*,state.focusorder] = $
                 [min(state.reg[0,*],MAX=max),max]
              findshifts_plotupdate
              findshifts_findshifts,state.focusorder
              findshifts_shiftspectra,state.focusorder
              
           end
           
           else:
           
        endcase
        findshifts_plotupdate,/MINMAX
        state.cursormode='None'
        
     endelse

  endif
          
;  Copy the pixmaps and draw the cross hair or zoom lines.
     
  wset, state.plotwin1_wid
  device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0,state.pixmap1_wid]
  
  wset, state.plotwin2_wid
  device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0,state.pixmap2_wid]

  wset, state.plotwin1_wid
  
  case state.cursormode of 
     
     'XZoom': begin

        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        wset, state.plotwin2_wid
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        
     end
        
     'YZoom': plots, [0,state.plotsize[0]],[event.y,event.y],COLOR=2,/DEVICE
     
     'Zoom': begin
        
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        plots, [0,state.plotsize[0]],[event.y,event.y],COLOR=2,/DEVICE
        xy = convert_coord(event.x,event.y,/DEVICE,/TO_DATA,/DOUBLE)
        plots,[state.reg[0,0],state.reg[0,0]],[state.reg[1,0],xy[1]],$
              LINESTYLE=2,COLOR=2
        plots,[state.reg[0,0],xy[0]],[state.reg[1,0],state.reg[1,0]],$
              LINESTYLE=2,COLOR=2
        wset, state.plotwin2_wid
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
                
     end

     'Select': begin

        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=7,/DEVICE
        wset, state.plotwin2_wid
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=7,/DEVICE
        
     end
     
     else: begin
        
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        plots, [0,state.plotsize[0]],[event.y,event.y],COLOR=2,/DEVICE
        wset, state.plotwin2_wid
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        
     end
     
  endcase
  
end
;
;===============================================================================
;
pro findshifts_plotwinevent2,event

  common findshifts_state

  widget_control, event.id,  GET_UVALUE=uvalue
  
  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin

     widget_control, state.plotwin2,INPUT_FOCUS=event.enter
     
     wset, state.plotwin1_wid
     device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0, $
                   state.pixmap1_wid]
     wset, state.plotwin2_wid
     device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0, $
                   state.pixmap2_wid]
     return
     
  endif

  ;  Check for arrow keys

  if event.type eq 6 and event.release eq 0 then begin

     case event.key of
              
        5: begin

           del = (state.wrange[1]-state.wrange[0])*0.3
           oldcen = (state.wrange[1]+state.wrange[0])/2.
           newcen = oldcen-del

           if newcen lt state.abswrange[0] then return
           state.wrange = state.wrange + (newcen-oldcen)
           findshifts_plotupdate,/MINMAX           
           
        end
        
        6: begin

           del = (state.wrange[1]-state.wrange[0])*0.3
           oldcen = (state.wrange[1]+state.wrange[0])/2.
           newcen = oldcen+del

           if newcen gt state.abswrange[1] then return
           state.wrange = state.wrange + (newcen-oldcen)
           findshifts_plotupdate,/MINMAX

        end
        
        else:
        
     endcase
     
  endif
     
  !p = state.pscale
  !x = state.xscale
  !y = state.yscale

;
;  Check for ASCII keyboard event
;
  if event.type eq 5 and event.release eq 1 then begin
  
     case strtrim(event.ch,2) of 

        'a': begin
           
           state.abswrange = state.wrange
           state.absyrange=state.yrange
           
        end
        
        'c': begin          
           
           state.cursormode = 'None'
           state.reg = !values.f_nan                
           findshifts_plotupdate
           
        end

        'd': begin

           state.focusorder = -1
           state.mode = 'All'
           findshifts_mkmenu,/DESTROY
           findshifts_plotupdate

        end
        
        'i': findshifts_zoom,/IN
        
        'o': findshifts_zoom,/OUT

        's': begin

           if state.focusorder eq -1 then begin

;  Find which order you are talking about              
              
              xydev = convert_coord(state.cenwave,replicate(1.0,state.norders),$
                                    /DATA,/TO_DEVICE)
              min = min(abs(reform(xydev[0,*])-event.x),z)
              state.focusorder = z              
              state.mode = 'Order'
              findshifts_mkmenu,/DESTROY
              findshifts_selectorder
              
           endif else begin
              
              state.corwranges[*,state.focusorder] = !values.f_nan
              findshifts_plotupdate,/MINMAX              

              state.cursormode = 'Select'
              state.reg = !values.f_nan
              
           endelse
                         
        end
        
        'w': begin
           
           state.wrange = state.abswrange
           state.yrange = state.absyrange
           findshifts_plotupdate,/MINMAX
           
        end
        
        'x': begin 
           
           state.cursormode = 'XZoom'
           state.reg = !values.f_nan
           
        end

        'y': begin 
           
           if state.plotshifts then return
           state.cursormode = 'YZoom'
           state.reg = !values.f_nan
           
        end
        
        'z': begin        

           if state.plotshifts then return
           state.cursormode = 'Zoom'
           state.reg = !values.f_nan
           
        end

        
        else:
        
     endcase
     return
     
  endif 

  wset, state.plotwin2_wid
  x  = event.x/float(state.plotsize[0])
  y  = event.y/float(state.plotsize[1])
  xy = convert_coord(x,y,/NORMAL,/TO_DATA,/DOUBLE)

  if event.type eq 1 then begin

     z = where(finite(state.reg) eq 1,count)
     if count eq 0 then begin
        
        wset, state.pixmap2_wid
        state.reg[*,0] = xy[0:1]
        case state.cursormode of
           
           'XZoom': begin

              plots, [event.x,event.x],[0,state.plotsize[1]], $
                     COLOR=2,/DEVICE,LINESTYLE=2

              wset, state.pixmap1_wid
              
              plots, [event.x,event.x],[0,state.plotsize[1]], $
                     COLOR=2,/DEVICE,LINESTYLE=2
              
           end
           
           'YZoom': plots, [0,state.plotsize[0]],[event.y,event.y], $
                           COLOR=2,/DEVICE,LINESTYLE=2

           'Select': begin

              plots, [event.x,event.x],[0,state.plotsize[1]], $
                     COLOR=7,/DEVICE,LINESTYLE=2
              
              wset, state.pixmap1_wid
              
              plots, [event.x,event.x],[0,state.plotsize[1]], $
                     COLOR=7,/DEVICE,LINESTYLE=2

           end
              
           else:
           
        endcase
        wset, state.plotwin1_wid
        device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0, $
                      state.pixmap1_wid]

        wset, state.plotwin2_wid
        device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0, $
                      state.pixmap2_wid]
        
     endif else begin 
        
        state.reg[*,1] = xy[0:1]
        case state.cursormode of 
           
           'XZoom': state.wrange = [min(state.reg[0,*],MAX=max),max]
           
           'YZoom': state.yrange = [min(state.reg[1,*],MAX=max),max]
           
           'Zoom': begin
              
              state.wrange = [min(state.reg[0,*],MAX=max),max]
              state.yrange = [min(state.reg[1,*],MAX=max),max]
              
           end

           'Select': begin

              state.corwranges[*,state.focusorder] = $
                 [min(state.reg[0,*],MAX=max),max]
              findshifts_findshifts,state.focusorder
              findshifts_shiftspectra,state.focusorder
              
           end
           
           else:
           
        endcase
        findshifts_plotupdate,/MINMAX
        state.cursormode='None'
        
     endelse

  endif
  
;  Copy the pixmaps and draw the cross hair or zoom lines.
     
  wset, state.plotwin1_wid
  device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0,state.pixmap1_wid]
  
  wset, state.plotwin2_wid
  device, COPY=[0,0,state.plotsize[0],state.plotsize[1],0,0,state.pixmap2_wid]

  wset, state.plotwin2_wid
  
  case state.cursormode of 
     
     'XZoom': begin

        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        wset, state.plotwin1_wid
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        
     end
        
     'YZoom': plots, [0,state.plotsize[0]],[event.y,event.y],COLOR=2,/DEVICE
     
     'Zoom': begin
        
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        plots, [0,state.plotsize[0]],[event.y,event.y],COLOR=2,/DEVICE
        xy = convert_coord(event.x,event.y,/DEVICE,/TO_DATA,/DOUBLE)
        plots,[state.reg[0,0],state.reg[0,0]],[state.reg[1,0],xy[1]],$
              LINESTYLE=2,COLOR=2
        plots,[state.reg[0,0],xy[0]],[state.reg[1,0],state.reg[1,0]],$
              LINESTYLE=2,COLOR=2
        wset, state.plotwin1_wid
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
                
     end

     'Select': begin

        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=7,/DEVICE
        wset, state.plotwin1_wid
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=7,/DEVICE
        
     end
     
     else: begin
        
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        plots, [0,state.plotsize[0]],[event.y,event.y],COLOR=2,/DEVICE
        wset, state.plotwin1_wid
        plots, [event.x,event.x],[0,state.plotsize[1]],COLOR=2,/DEVICE
        
     end
     
  endcase


  
end
;
;===============================================================================
;
pro findshifts_resizeevent, event

  common findshifts_state

  widget_control, state.findshifts_base, TLB_GET_SIZE=size

  xsize = size[0]
  ysize = size[1]

  state.plotsize[1] = (ysize-state.winbuffer[1])/2
  state.plotsize[0] = xsize-state.winbuffer[0]

  widget_control, state.findshifts_base,UPDATE=0
  
  widget_control, state.plotwin1, DRAW_XSIZE=state.plotsize[0], $
                  DRAW_YSIZE=state.plotsize[1]

  widget_control, state.plotwin2, DRAW_XSIZE=state.plotsize[0], $
                  DRAW_YSIZE=state.plotsize[1]           

  widget_control, state.findshifts_base,UPDATE=1
  
  widget_geom = widget_info(state.findshifts_base, /GEOMETRY)
  
  state.winbuffer[0]=widget_geom.xsize-state.plotsize[0]
  state.winbuffer[1]=widget_geom.ysize-2*state.plotsize[1]
  
  
  wdelete,state.pixmap1_wid
  window, /FREE, /PIXMAP,XSIZE=state.plotsize[0],YSIZE=state.plotsize[1]
  state.pixmap1_wid = !d.window

  wdelete,state.pixmap2_wid
  window, /FREE, /PIXMAP,XSIZE=state.plotsize[0],YSIZE=state.plotsize[1]
  state.pixmap2_wid = !d.window  
  
  findshifts_plotupdate
  
end
;
;===============================================================================
;
; ------------------------------Main Program-------------------------------- 
;
;===============================================================================
;
function xmc_findshifts,objspec,telspec,orders,awave,atrans,xtitle, $
                        CANCEL=cancel

  cancel = 0

  mc_mkct
  common findshifts_state

  if not xregistered('xmc_findshifts') then begin
     
     findshifts_initcommon,objspec,telspec,orders,awave,atrans,xtitle, $
                           CANCEL=cancel

     if cancel then return,-1
     
     if keyword_set(PARENT) ne 0 then widget_control, parent,SENSITIVE=0
     
     state.findshifts_base = widget_base(TITLE='Findshifts', $
                                        /COLUMN,$
                                        /TLB_SIZE_EVENTS)

     button = widget_button(state.findshifts_base,$
                            FONT=state.buttonfont,$
                            EVENT_PRO='findshifts_event',$
                            VALUE='Cancel',$
                            UVALUE='Cancel')
     
        state.menu = widget_base(state.findshifts_base,$
                                 /ROW,$
                                 FRAME=2,$
                                 /BASE_ALIGN_CENTER,$
                                 EVENT_PRO='findshifts_event')

        if state.naps gt 1 then begin

           values = strtrim(string(indgen(state.naps)+1,FORMAT='(I2)'),2)
           
           state.selectap_dl = widget_droplist(state.menu,$
                                               FONT=state.buttonfont,$
                                               TITLE='Ap: ',$
                                               VALUE=values,$
                                               UVALUE='Select Ap Button')


        Endif
        
        Label = widget_label(state.menu,$
                             VALUE='Plot:',$
                             FONT=state.buttonfont)
                
           subrow = widget_base(state.menu,$
                                /ROW,$
                                /BASE_ALIGN_CENTER,$
                                /TOOLBAR,$
                                /NONEXCLUSIVE)

              button = widget_button(subrow,$
                                     FONT=state.buttonfont,$
                                     VALUE='Atmosphere',$
                                     UVALUE='Plot Atmosphere Button')
              widget_control, button, SET_BUTTON=state.plotatmosphere

              subrow = widget_base(state.menu,$
                                   /ROW,$
                                   /TOOLBAR,$
                                   /EXCLUSIVE)

                 button = widget_button(subrow, $
                                        VALUE='2 Color', $
                                        EVENT_PRO='findshifts_event',$
                                        UVALUE='2 Color Button',$
                                        FONT=state.buttonfont)
                 if state.altcolor eq 2 then widget_control, button,/SET_BUTTON
              
                 button = widget_button(subrow, $
                                        VALUE='3 Color', $
                                        EVENT_PRO='findshifts_event',$
                                        UVALUE='3 Color Button',$
                                        FONT=state.buttonfont)
                 if state.altcolor eq 3 then widget_control, button,/SET_BUTTON
                            
;           subrow = widget_base(state.menu,$
;                                /ROW,$
;                                /BASE_ALIGN_CENTER,$
;                                /TOOLBAR,$
;                                /EXCLUSIVE)
              
        state.plotwin1 = widget_draw(state.findshifts_base,$
                                     /ALIGN_CENTER,$
                                     XSIZE=state.plotsize[0],$
                                     YSIZE=state.plotsize[1],$
                                     EVENT_PRO='findshifts_plotwinevent1',$
                                     /MOTION_EVENTS,$
                                     /KEYBOARD_EVENTS,$
                                     /BUTTON_EVENTS,$
                                     /TRACKING_EVENTS)

        row = widget_base(state.findshifts_base,$
                          /ROW,$
                          FRAME=2)

           label = widget_label(row,$
                                VALUE='Plot:',$
                                FONT=state.buttonfont)
           
           subrow = widget_base(row,$
                                EVENT_PRO='findshifts_event',$
                                /ROW,$
                                /BASE_ALIGN_CENTER,$
                                /TOOLBAR,$
                                /EXCLUSIVE)
       
              state.pltshifts_but = widget_button(subrow,$
                                                  FONT=state.buttonfont,$
                                                  VALUE='Shifts',$
                                                  UVALUE='Plot Shifts Button')
              widget_control, state.pltshifts_but, SET_BUTTON=state.plotshifts
           
              state.pltspecs_but = widget_button(subrow,$
                                                 FONT=state.buttonfont,$
                                                 VALUE='Shifted Spectra',$
                                          UVALUE='Plot Shifted Spectra Button')
              widget_control, state.pltspecs_but, $
                              SET_BUTTON=state.plotshiftedspectra

                     
        state.plotwin2 = widget_draw(state.findshifts_base,$
                                     /ALIGN_CENTER,$
                                     XSIZE=state.plotsize[0],$
                                     YSIZE=state.plotsize[1],$
                                     EVENT_PRO='findshifts_plotwinevent2',$
                                     /MOTION_EVENTS,$
                                     /KEYBOARD_EVENTS,$
                                     /BUTTON_EVENTS,$
                                     /TRACKING_EVENTS)


     row = widget_base(state.findshifts_base,$
                       /ROW,$
                       /BASE_ALIGN_LEFT,$
                       FRAME=2)

        xmin = coyote_field2(row,$
                             LABELFONT=state.buttonfont,$
                             FIELDFONT=state.textfont,$
                             TITLE='X Min:',$
                             UVALUE='X Min',$
                             XSIZE=12,$
                             EVENT_PRO='findshifts_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.xmin_fld = [xmin,textid]
                
        xmax = coyote_field2(row,$
                             LABELFONT=state.buttonfont,$
                             FIELDFONT=state.textfont,$
                             TITLE='X Max:',$
                             UVALUE='X Max',$
                             XSIZE=12,$
                             EVENT_PRO='findshifts_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.xmax_fld = [xmax,textid]
        
        ymin = coyote_field2(row,$
                             LABELFONT=state.buttonfont,$
                             FIELDFONT=state.textfont,$
                             TITLE='Y Min:',$
                             UVALUE='Y Min',$
                             XSIZE=12,$
                             EVENT_PRO='findshifts_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.ymin_fld = [ymin,textid]
        
        ymax = coyote_field2(row,$
                             LABELFONT=state.buttonfont,$
                             FIELDFONT=state.textfont,$
                             TITLE='Y Max:',$
                             UVALUE='Y Max',$
                             XSIZE=12,$
                             EVENT_PRO='findshifts_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.ymax_fld = [ymax,textid]

     state.slider = widget_slider(state.findshifts_base,$
                                  UVALUE='Slider',$
                                  EVENT_PRO='findshifts_event',$
                                  /DRAG,$
                                  /SUPPRESS_VALUE,$
                                  FONT=buttonfont)
     widget_control, state.slider, SET_VALUE=state.sliderval
           
     button = widget_button(state.findshifts_base,$
                            FONT=state.buttonfont,$
                            EVENT_PRO='findshifts_event',$
                            VALUE='Accept',$
                            UVALUE='Accept')

     findshifts_mkmenu
            
; Get things running.  Center the widget using the Fanning routine.
     
     cgcentertlb,state.findshifts_base
     widget_control, state.findshifts_base, /REALIZE     

;  Get plotwin ids

   widget_control, state.plotwin1, GET_VALUE=x
   state.plotwin1_wid=x
   window, /FREE, /PIXMAP,XSIZE=state.plotsize[0],YSIZE=state.plotsize[1]
   state.pixmap1_wid = !d.window

   widget_control, state.plotwin2, GET_VALUE=x
   state.plotwin2_wid=x
   window, /FREE, /PIXMAP,XSIZE=state.plotsize[0],YSIZE=state.plotsize[1]
   state.pixmap2_wid = !d.window
   
;  Get sizes for things.
   
   widget_geom = widget_info(state.findshifts_base, /GEOMETRY)

   state.winbuffer[0]=widget_geom.xsize-state.plotsize[0]
   state.winbuffer[1]=widget_geom.ysize-2*state.plotsize[1]
   
   findshifts_plotupdate,/MINMAX
     
; Start the Event Loop. This will be a non-blocking program.
     
     XManager, 'xmc_findshifts', $
               state.findshifts_base, $
               EVENT_HANDLER='findshifts_resizeevent'
        
     if n_elements(PARENT) ne 0 then widget_control, parent,SENSITIVE=1
     
     cancel = state.cancel
     shifts = state.shifts
     z = where(finite(shifts) eq 0,cnt)
     if cnt ne 0 then shifts[z] = 0.0

     state = 0B     
     return, shifts


  endif


end
