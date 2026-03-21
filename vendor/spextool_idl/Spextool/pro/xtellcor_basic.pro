;+
; NAME:
;     xtellcor_basic
;
; PURPOSE:
;     Telluric corrects SpeX spectra by simply dividing by the std star.
;    
; CATEGORY:
;     Widget
;
; CALLING SEQUENCE:
;     xtellcor_basic
;    
; INPUTS:
;     None
;    
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     None
;     
; OUTPUTS:
;     Write spectral FITS file to disk
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
;     Divide by standard and multiple by blackbody
;
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;     2002 - Written by M. Cushing, Institute for Astronomy, UH
;-
;
;===============================================================================
;
; ------------------------------Event Handlers-------------------------------- 
;
;===============================================================================
;
pro xtellcorbasic_event,event

  widget_control, event.id,  GET_UVALUE = uvalue

  if uvalue eq 'Quit' then begin
     
     widget_control, event.top, /DESTROY
     goto, getout
     
  endif
    
  widget_control, event.top, GET_UVALUE = state, /NO_COPY
  widget_control, /HOURGLASS
  
  case uvalue of

;     'Additional Output': begin
;        
;        if event.value eq 'Telluric' then state.telluricoutput=event.select
;        
;     end
     
     'Write File': xtellcorbasic_writefile,state
     
     'Get Shifts': xtellcorbasic_getshifts,state

     'Help': begin
        
        pre = (strupcase(!version.os_family) eq 'WINDOWS') ? 'start ':'open '
        
        spawn, pre+filepath('spextoolmanual.pdf', $
                            ROOT=state.spextoolpath,$
                            SUBDIR='manuals')
        
     end
     
     'Load Spectra Button': xtellcorbasic_loadspec,state
     
     'Object Spectra Button': begin
        
        obj = dialog_pickfile(DIALOG_PARENT=state.xtellcorbasic_base,$
                              PATH=state.objpath,/MUST_EXIST, $
                              GET_PATH=path,FILTER='*.fits')
        if obj eq '' then goto, cont
        widget_control,state.objspectra_fld[1],SET_VALUE=file_basename(obj)
        state.objpath = path
        mc_setfocus,state.objspectra_fld
        
     end
       
     'Restore Continuum': begin
        
        state.restorecont = event.value
        widget_control, state.temp_fld[0],SENSITIVE=event.value
        widget_control, state.vmag_fld[0],SENSITIVE=event.value
        widget_control, state.units_dl,SENSITIVE=event.value
        
     end
     
     'Shift Object Ap': begin

        state.shiftobjap = event.index
        z = where(*state.objorders eq state.shiftobjorder)

     end

     'Spectrum Units': begin
        
        case event.index of 
           
           0: state.units = 'ergs s-1 cm-2 A-1'
           1: state.units = 'ergs s-1 cm-2 Hz-1'
           2: state.units = 'W m-2 um-1'
           3: state.units = 'W m-2 Hz-1'
           4: state.units = 'Jy'
           
        endcase 
        
     end
     
     'Standard Spectra Button': begin
        
        std = dialog_pickfile(DIALOG_PARENT=state.xtellcorbasic_base,$
                              PATH=state.stdpath,/MUST_EXIST, $
                              GET_PATH=path,FILTER='*.fits')
        if std eq '' then goto, cont
        widget_control,state.stdspectra_fld[1],SET_VALUE=file_basename(std)
        state.stdpath = path
        mc_setfocus,state.stdspectra_fld
        
     end
     
  endcase
  
;  Put state variable into the user value of the top level base.
  
  cont: 
  widget_control, state.xtellcorbasic_base, SET_UVALUE=state, /NO_COPY
  getout:

end
;
;===============================================================================
;
; ----------------------------Support procedures------------------------------ 
;
;===============================================================================
;
pro xtellcorbasic_changeunits,wave,spec,spec_error,state

  c = 2.99792458e+8
  ang = '!5!sA!r!u!9 %!5!n'
  
  case state.units of 
     
     'ergs s-1 cm-2 A-1': state.nytitle = $
        '!5f!D!7k!N!5 (ergs s!E-1!N cm!E-2!N '+ang+'!E-1!N)' 
     
     'ergs s-1 cm-2 Hz-1': begin
        
        spec            = temporary(spec)* wave^2 * (1.0e-2 / c)
        spec_error      = temporary(spec_error)* wave^2 * (1.0e-2 / c)
        state.nytitle = '!5f!D!7m!N!5 (ergs s!E-1!N cm!E-2!N Hz!E-1!N)'
        
     end
     'W m-2 um-1': begin 
        
        spec            = temporary(spec)*10.
        spec_error      = temporary(spec_error)*10.
        state.nytitle = '!5f!D!7k!N!5 (W m!E-2!N !7l!5m!E-1!N)'
        
     end
     'W m-2 Hz-1': begin
        
        spec            = temporary(spec)* wave^2 * (1.0e-5 / c)
        spec_error      = temporary(spec_error)* wave^2 * (1.0e-5 / c)
        state.nytitle = '!5f!D!7m!N!5 (W m!E-2!N Hz!E-1!N)' 
        
     end
     
     'Jy': begin
        
        spec            = temporary(spec)* wave^2 * (1.0e21 / c) 
        spec_error      = temporary(spec_error)* wave^2 * (1.0e21 / c) 
        state.nytitle = '!5f!D!7m!N!5 (Jy)' 
        
     end
     
  endcase
  
end
;
;===============================================================================
;
pro xtellcorbasic_cleanup,base

widget_control, base, GET_UVALUE = state, /NO_COPY
if n_elements(state) ne 0 then begin

   ptr_free, state.awave
   ptr_free, state.atrans
   ptr_free, state.obj
   ptr_free, state.objhdr
   ptr_free, state.objorders
   ptr_free, state.shifts
   ptr_free, state.stdhdr
   ptr_free, state.stdorders
   ptr_free, state.tel

endif
state = 0B

end
;
;===============================================================================
;
pro xtellcorbasic_getshifts,state

  norders = n_elements(*state.objorders)
  idx = indgen(norders)*state.objnaps+state.shiftobjap

  shifts = xmc_findshifts((*state.obj)[*,*,idx],*state.tel,*state.objorders, $
                          *state.awave,*state.atrans,state.xtitle,CANCEL=cancel)

  if not cancel then (*state.shifts)[*,state.shiftobjap] = shifts

end
;
;===============================================================================
;
pro xtellcorbasic_loadspec,state

;  Get files.

  stdfile = mc_cfld(state.stdspectra_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  stdfile = mc_cfile(state.stdpath+stdfile, $
                     WIDGET_ID=state.xtellcorbasic_base,$
                     CANCEL=cancel)
  if cancel then return
  
  objfile = mc_cfld(state.objspectra_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  objfile = mc_cfile(state.stdpath+objfile, $
                     WIDGET_ID=state.xtellcorbasic_base,CANCEL=cancel)
  if cancel then return

  if state.restorecont then begin

     temp = mc_cfld(state.temp_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     
     vmag = mc_cfld(state.vmag_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     
  endif
  
;  Read files
  
  mc_readspec,stdfile,std,stdhdr,stdobsmode,start,stop,stdnorders,stdnaps,$
              stdorders,stdxunits,stdyunits,slith_pix,slith_arc,slitw_pix, $
              slitw_arc,stdrp,stdairmass,stdxtitle,CANCEL=cancel
  if cancel then return
  
  mc_readspec,objfile,obj,objhdr,objobsmode,start,stop,objnorders,objnaps,$
              objorders,objxunits,objyunits,slith_pix,slith_arc,slitw_pix, $
              slitw_arc,objrp,objairmass,CANCEL=cancel
  if cancel then return

;  Construct the telluric correction spectra

  tel = obj
  for i = 0,objnaps-1 do begin
     
     for j = 0,objnorders-1 do begin
        
        k = j*objnaps+i
        z = where(stdorders eq objorders[j],cnt)
        if cnt eq 0 then continue
        
;  Interpolate telluric spectrum onto object wavelength sampling.
        
        mc_interpspec,std[*,0,z],std[*,1,z],obj[*,0,k],std_flux,std_error,$
                      IYERROR=std[*,2,z],CANCEL=cancel
        if cancel then return
        
        tellspec = 1.0/std_flux
        tellunc  = (1.0/std_flux^2) * std_error
        
        if state.restorecont then begin
           
;  Flux calibrate the Blackbody
           
           bbflux_v = planck(5556,temp)
           scale = ( 3.46e-9*10^(-0.4*(vmag-0.03)) )/bbflux_v
           bbflux = planck(obj[*,0,k]*10000.,temp)*scale
           
           tellspec = temporary(tellspec) * bbflux
           tellunc  = temporary(tellunc) * bbflux
           
        endif
        
        xtellcorbasic_changeunits,obj[*,0,k],tellspec,tellunc,state
        
        tel[*,1,k] = tellspec
        tel[*,2,k] = tellunc

     endfor
     
  endfor  
  
;  Store results and zero things
  
  *state.stdorders = stdorders
  *state.objorders = objorders
  state.objnaps    = objnaps
  *state.tel       = tel
  *state.obj       = obj
  *state.objhdr    = objhdr
  *state.stdhdr    = stdhdr
  state.xtitle     = stdxtitle
  
  state.shiftobjap = 0
  *state.shifts = fltarr(objnorders,objnaps)

;  Check the airmass
  
  if abs(stdairmass-objairmass) gt 0.1 then begin

     color = 17
     beep
     beep
     beep

  endif else color = 16

  wset, state.message
  erase,COLOR=color
  xyouts,10,8,'Std Airmass:'+string(stdairmass,FORMAT='(f7.4)')+ $
         ', Obj Airmass:'+string(objairmass,FORMAT='(f7.4)')+ $
         ', (Std-Obj) Airmass: '+ $
         string((stdairmass-objairmass),FORMAT='(f7.4)'),/DEVICE,$
         CHARSIZE=1.1,FONT=0
  
  widget_control, state.objap_dl,$
                  SET_VALUE=string(indgen(objnaps)+1,FORMAT='(i2.2)')

;  Get the atmospheric transmission

  files = file_basename(file_search(filepath('atran*.fits', $
                                             ROOT_DIR=state.spextoolpath, $
                                             SUBDIR='data')))
  
  nfiles = n_elements(files)
  rps = lonarr(nfiles)
  for i =0,nfiles-1 do rps[i] = long(strmid(file_basename(files[i],'.fits'),5))
  
  min = min(abs(rps-objrp),idx)
  
  spec = readfits(filepath('atran'+strtrim(rps[idx],2)+'.fits', $
                           ROOT_DIR=state.spextoolpath,SUBDIR='data'),/SILENT)
  *state.awave = reform(spec[*,0])
  *state.atrans = reform(spec[*,1])

end
;
;==============================================================================
;
pro xtellcorbasic_writefile,state
  
  tel = *state.tel
  obj = *state.obj
  
  stdfile = mc_cfld(state.stdspectra_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  oname = mc_cfld(state.objoname_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  if state.restorecont then begin
     
     temp = mc_cfld(state.temp_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     
     vmag = mc_cfld(state.vmag_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     
  endif 

  for i = 0,state.objnaps-1 do begin
     
     for j = 0,n_elements(*state.objorders)-1 do begin
        
        k = j*state.objnaps+i
        z = where(*state.stdorders eq (*state.objorders)[j],cnt)
        if cnt eq 0 then continue
        
        tellspec  = tel[*,1,j]
        tellunc = tel[*,2,j]
        
;  Shift the spectrum
        
        x = findgen(n_elements(tellspec))
        mc_interpspec,x+(*state.shifts)[j,i],tellspec,x,stellspec,$
                   stellunc,IYERROR=tellunc,CANCEL=cancel
        if cancel then return

        corspec    = obj[*,1,k]*stellspec
        corspecunc = sqrt(stellspec^2 * obj[*,2,k]^2+obj[*,1,k]^2 * stellunc^2)
        
        obj[*,1,k] = corspec
        obj[*,2,k] = corspecunc

     endfor
     
     cont:
     
  endfor
  
;  Get hdr and history down

  history = 'The spectra were divided by the spectra of the standard star '+$
            strtrim(stdfile,2)+'.'
    
  if state.restorecont then begin
          
     history = history+'  The spectra were also multiplied by a  '+$
               'blackbody with a temperature of '+strtrim(temp,2)+' K ' + $
               'scaled to a V-band magnitude of' + $
               ' '+string(vmag,FORMAT='(f6.3)')+'.'
     
     yunits = state.units
     ytitle = state.nytitle
     
  endif else begin 

     yunits = 'None'
     ytitle = '!5Ratio'
     history = history+' Therefore the spectrum is unitless.'

  endelse
  
;  Write the corrected spectrum to disk.

  fxhmake,newhdr,obj
  
  hdrinfo = mc_gethdrinfo(*state.objhdr,state.keywords,/IGNOREMISSING, $
                          CANCEL=cancel)
  if cancel then return

  
  ntags = n_tags(hdrinfo.vals)
  names = tag_names(hdrinfo.vals)

  for i = 0, ntags-1 do begin

     if names[i] eq 'HISTORY' then begin

        sxaddhist,(hdrinfo.vals.(i)),newhdr
        continue

     endif
     
     for k = 0,n_elements(hdrinfo.vals.(i))-1 do begin
        
        if size((hdrinfo.vals.(i))[k],/TYPE) eq 7 and $
           strlen((hdrinfo.vals.(i))[k]) gt 68 then begin
           
           fxaddpar,newhdr,names[i],(hdrinfo.vals.(i))[k],(hdrinfo.coms.(i))[k]
           
        endif else sxaddpar,newhdr,names[i],(hdrinfo.vals.(i))[k], $
                            (hdrinfo.coms.(i))[k]
        
     endfor
     
  endfor
  
  sxaddpar,newhdr,'FILENAME',strtrim(oname+'.fits',2)
  sxaddpar,newhdr,'YUNITS',yunits, 'Units of the Y axis'
  sxaddpar,newhdr,'YTITLE',ytitle, 'IDL Y Title'

  sxaddhist,' ',newhdr
  sxaddhist,'###################### Xtelcorr Basic History ' + $
            '######################',newhdr
  sxaddhist,' ',newhdr
  
  history = mc_splittext(history,67,CANCEL=cancel)
  if cancel then return
  sxaddhist,history,newhdr

  
  writefits, state.objpath+oname+'.fits',obj,newhdr
  xvspec,state.objpath+oname+'.fits'

end
;
;===============================================================================
;
; ------------------------------Main Program---------------------------------- 
;
;===============================================================================
;
;
pro xtellcor_basic

;  Get spextool and instrument information 
  
  mc_getspextoolinfo,spextoolpath,packagepath,spextool_keywords,instrinfo, $
                     notirtf,version,CANCEL=cancel
  if cancel then return
  
;  Set the fonts

  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return

  state = {awave:ptr_new(2),$
           atrans:ptr_new(2),$
           bmag_fld:[0L,0L],$
           instrument:instrinfo.instrument,$
           keywords:[spextool_keywords,instrinfo.xtellcor_keywords,'HISTORY'],$
           message:0L,$
           nytitle:'',$
           obj:ptr_new(fltarr(2)),$
           objap_dl:0L,$
           objhdr:ptr_new(fltarr(2)),$
           objoname_fld:[0L,0L],$
           objorders:ptr_new(fltarr(2)),$
           objnaps:0,$
           objpath:'',$
           objspectra_fld:[0L,0L],$
           path_fld:[0L,0L],$
           restorecont:1,$
           shift:0L,$
           shiftobjap:0,$
           shifts:ptr_new(2),$
           spextoolpath:spextoolpath,$
           stdhdr:ptr_new(fltarr(2)),$
           stdorders:ptr_new(fltarr(2)),$
           stdpath:'',$
           stdspectra_fld:[0L,0L],$
           tel:ptr_new(fltarr(2)),$
           temp:0.,$
           temp_fld:[0L,0L],$
;           telluricoutput:0,$
           xtellcorbasic_base:0L,$
           xtitle:'',$
           units:'ergs s-1 cm-2 A-1',$
           units_dl:0L,$
           userkeywords:instrinfo.xtellcor_keywords,$
           vmag_fld:[0L,0L]}
  
  state.xtellcorbasic_base = widget_base(TITLE='Xtellcor (Basic)', $
                                         EVENT_PRO='xtellcorbasic_event',$
                                         /COLUMN)
  
     button = widget_button(state.xtellcorbasic_base,$
                            FONT=buttonfont,$
                            VALUE='Done',$
                            UVALUE='Quit')

     message = widget_draw(state.xtellcorbasic_base,$
                           FRAME=2,$
                           XSIZE=10,$
                           YSIZE=25)
         
     row_base = widget_base(state.xtellcorbasic_base,$
                            /ROW)

        col1_base = widget_base(row_base,$
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
              
                 but = widget_button(row,$
                                     FONT=buttonfont,$
                                     VALUE='Std Spectra',$
                                     UVALUE='Standard Spectra Button')
                 
                 std_fld = coyote_field2(row,$
                                         LABELFONT=buttonfont,$
                                         FIELDFONT=textfont,$
                                         TITLE=':',$
                                         UVALUE='Standard Spectra Field',$
                                         VALUE='cspectra51-60.fits',$
                                         XSIZE=18,$
                                         /CR_ONLY,$
                                         TEXTID=textid)
                 state.stdspectra_fld = [std_fld,textid]

              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 but = widget_button(row,$
                                     FONT=buttonfont,$
                                     VALUE='Obj Spectra',$
                                     UVALUE='Object Spectra Button')
                 
                 obj_fld = coyote_field2(row,$
                                         LABELFONT=buttonfont,$
                                         FIELDFONT=textfont,$
                                         TITLE=':',$
                                         UVALUE='Object Spectra Field',$
                                         VALUE='cspectra61-70.fits',$
                                         XSIZE=18,$
                                         /CR_ONLY,$
                                         TEXTID=textid)
                 state.objspectra_fld = [obj_fld,textid]

              convolve_bg = cw_bgroup(box1_base,$
                                      FONT=buttonfont,$
                                      ['No','Yes'],$
                                      /ROW,$
                                      /RETURN_INDEX,$
                                      /NO_RELEASE,$
                                      /EXCLUSIVE,$
                                      LABEL_LEFT='Restore Continuum:',$
                                      UVALUE='Restore Continuum',$
                                      SET_VALUE=state.restorecont)
              
              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE='BB Temp (K):',$
                                     UVALUE='BB Temp Field',$
                                     XSIZE=5,$
                                     TEXTID=textid)
                 state.temp_fld = [fld,textid]         
                 
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE='V Mag:',$
                                     UVALUE='V Mag Field',$
                                     XSIZE=5,$
                                     TEXTID=textid)
                 state.vmag_fld = [fld,textid]         
                 
              value =['ergs s-1 cm-2 A-1','ergs s-1 cm-2 Hz-1',$
                      'W m-2 um-1','W m-2 Hz-1','Jy']

              state.units_dl = widget_droplist(box1_base,$
                                               FONT=buttonfont,$
                                               TITLE='Units:',$
                                               VALUE=value,$
                                               UVALUE='Spectrum Units')









                 
              button = widget_button(box1_base,$
                                     FONT=buttonfont,$
                                     VALUE='Construct Telluric Spectra',$
                                     UVALUE='Load Spectra Button')
              
              
        col2_base = widget_base(row_base,$
                                /COLUMN)

           box2_base = widget_base(col2_base,$
                                   /COLUMN,$
                                   FRAME=2)
        
              label = widget_label(box2_base,$
                                   VALUE='2.  Determine Shift',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)
              
              row = widget_base(box2_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 state.objap_dl = widget_droplist(row,$
                                                    FONT=buttonfont,$
                                                    TITLE='Aperture:',$
                                                    VALUE='01',$
                                                    UVALUE='Shift Object Ap')
                 
              shift = widget_button(box2_base,$
                                    VALUE='Get Shifts',$
                                    UVALUE='Get Shifts',$
                                    FONT=buttonfont)

        
           box3_base = widget_base(col2_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box3_base,$
                                   VALUE='3.  Write File',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)
              
              oname = coyote_field2(box3_base,$
                                    LABELFONT=buttonfont,$
                                    FIELDFONT=textfont,$
                                    TITLE='Object File:',$
                                    UVALUE='Object File',$
                                    XSIZE=18,$
                                    TEXTID=textid)
              state.objoname_fld = [oname,textid]

;              bg = cw_bgroup(box3_base,$
;                             FONT=buttonfont,$
;                             ['Telluric'],$
;                             /ROW,$
;                             /RETURN_NAME,$
;                             /NONEXCLUSIVE,$
;                             LABEL_LEFT='Additional:',$
;                             UVALUE='Additional Output')
;              
              button = widget_button(box3_base,$
                                     VALUE='Write File',$
                                     UVALUE='Write File',$
                                     FONT=buttonfont)
              
           help = widget_button(state.xtellcorbasic_base,$
                                VALUE='Help',$
                                UVALUE='Help',$
                                FONT=buttonfont)

      
; Get things running.  Center the widget using the Fanning routine.

   mc_mkct            
   cgcentertlb,state.xtellcorbasic_base
   widget_control, state.xtellcorbasic_base, /REALIZE

  widget_control, state.xtellcorbasic_base, /REALIZE
  widget_control, message, GET_VALUE=x
  state.message = x
  wset, x
  erase, color=196

  widget_geom = widget_info(state.xtellcorbasic_base, /GEOMETRY)
  widget_control, message, XSIZE=widget_geom.xsize-17
  erase, color=196   
   
; Start the Event Loop. This will be a non-blocking program.

   XManager, 'xtellcorbasic', $
             state.xtellcorbasic_base, $
             CLEANUP='xtellcorbasic_cleanup',$
             /NO_BLOCK
   
   widget_control, state.xtellcorbasic_base, SET_UVALUE=state, /NO_COPY
   

end
