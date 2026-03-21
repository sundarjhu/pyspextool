;+
; NAME:
;     xtellcor
;    
; PURPOSE:
;     Runs the SpeX telluric correction.
;    
; CATEGORY:
;     Widget
;
; CALLING SEQUENCE:
;     xtellcor
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
;     Writes a SpeX spectral FITS file to disk of the telluric
;     corrected spectra and optional the telluric correction spectrum
;     and the convolved Vega model.
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
;     Follow the directions
;
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;     2001 - Written by M. Cushing, Institute for Astonomy, UH
;     2008-02-13 - Removed output as a text file as an option.
;-
;
;===============================================================================
;
; ------------------------------Event Handlers-------------------------------- 
;
;===============================================================================
;
pro xtellcor_event, event

  widget_control, event.id,  GET_UVALUE=uvalue
  if uvalue eq 'Quit' then begin
     
     widget_control, event.top, /DESTROY
     goto, getout
     
  endif
  
  
  widget_control, event.top, GET_UVALUE=state, /NO_COPY
  widget_control, /HOURGLASS
  
  case uvalue of
     
     'Additional Output': begin
        
        if event.value eq 'Telluric' then state.telluricoutput=event.select
        if event.value eq 'Model' then state.vegaoutput=event.select
        
     end
     
     'B Magnitude Field': mc_setfocus, state.vmag_fld
     
     'Construct Kernel': xtellcor_conkernel,state
     
     'Construct Telluric Spectra': xtellcor_contellspec,state
     
     'Get Shifts': xtellcor_getshifts,state
     
     'Help': begin

        pre = (strupcase(!version.os_family) eq 'WINDOWS') ? 'start ':'open '
        
        spawn, pre+filepath(strlowcase(state.instrument)+'_spextoolmanual.pdf',$
                            ROOT=state.packagepath,$
                            SUBDIR='manual')
        
     end
          
     'Load Spectra': xtellcor_loadspec,state
     
     'Method': begin
        
        state.method = event.value
        sensitive=(event.value eq 'IP') ? 0:1
        widget_control, state.stdorder_dl,SENSITIVE=sensitive
        
     end
     
     'Object Spectra Button': begin
        
        obj = dialog_pickfile(DIALOG_PARENT=state.xtellcor_base,$
                              PATH=state.objpath,/MUST_EXIST, $
                              GET_PATH=path,FILTER='*.fits')
        if obj eq '' then goto, cont
        widget_control,state.objspectra_fld[1],SET_VALUE=file_basename(obj)
        state.objpath = path
        mc_setfocus,state.objspectra_fld
        
     end
     
     'Plot Object Ap': begin
        
        if state.continue lt 4 then begin
           
           ok = dialog_message('Previous steps not complete.',/ERROR,$
                               DIALOG_PARENT=state.xtellcor_base)
           goto, cont
           
        endif
        
        state.shiftobjap = event.index
        xtellcor_getshifts,state
        
     end
     
     'Scale Lines': xtellcor_getscales,state
     
     'Spectrum Units': begin
        
        case event.index of 
           
           0: state.units = 'ergs s-1 cm-2 A-1'
           1: state.units = 'ergs s-1 cm-2 Hz-1'
           2: state.units = 'W m-2 um-1'
           3: state.units = 'W m-2 Hz-1'
           4: state.units = 'Jy'
           
        endcase 
        
     end
     
     'Standard Order': state.stdorder = (*state.stdorders)[event.index]
     
     'Standard Spectra Button': begin
        
        std = dialog_pickfile(DIALOG_PARENT=state.xtellcor_base,$
                              PATH=state.stdpath,/MUST_EXIST, $
                              GET_PATH=path,FILTER='*.fits')
        
        if std eq '' then goto, cont
        widget_control,state.stdspectra_fld[1],SET_VALUE = file_basename(std)
        state.stdpath = path
        mc_setfocus,state.stdspectra_fld
        
     end
     
     'Standard Spectra Field': mc_setfocus,state.bmag_fld
     
     'V Magnitude Field': mc_setfocus, state.objspectra_fld
     
     'Write File': xtellcor_writefile,state
     
     else:
     
  endcase
  
;  Put state variable into the user value of the top level base.
   
  cont: 
  widget_control, state.xtellcor_base, SET_UVALUE=state, /NO_COPY
  getout:
  
end
;
;===============================================================================
;
; ----------------------------Support procedures------------------------------ 
;
;===============================================================================
;
pro xtellcor_cleanup,base

  widget_control, base, GET_UVALUE = state, /NO_COPY
  if n_elements(state) ne 0 then begin
     
     ptr_free, state.scales
     ptr_free, state.shift
     ptr_free, state.kernels
     ptr_free, state.objhdr
     ptr_free, state.objorders
     ptr_free, state.objspec
     ptr_free, state.nstd
     ptr_free, state.stddisp
     ptr_free, state.stdfwhm
     ptr_free, state.stdhdr
     ptr_free, state.stdorders
     ptr_free, state.stdspec
     ptr_free, state.tellspec
     ptr_free, state.fvega
     ptr_free, state.wvega
     ptr_free, state.varrinfo
     ptr_free, state.vegaspec
     ptr_free, state.cfvega
     ptr_free, state.cf2vega
     
     cutreg = *state.cutreg
     for i = 0, n_tags(cutreg)-1 do ptr_free, cutreg.(i)
     
     ptr_free, state.cutreg
     
     
  endif
  state = 0B
  
end
;
;===============================================================================
;
pro xtellcor_changeunits,wave,tellspec,tellspec_error,scvega,state

  ang = '!5!sA!r!u!9 %!5!n'
  case state.units of 
     
     'ergs s-1 cm-2 A-1': state.nytitle = $
        '!5f!D!7k!N!5 (ergs s!E-1!N cm!E-2!N '+ang+'!E-1!N'
     
     'ergs s-1 cm-2 Hz-1': begin
        
        tellspec = mc_chfunits(wave,tellspec,0,1,3,IERROR=tellspec_error, $
                               OERROR=tellspec_error)
        scvega = mc_chfunits(wave,scvega,0,1,3)
        
        state.nytitle = '!5f!D!7m!N!5 (ergs s!E-1!N cm!E-2!N Hz!E-1!N'
        
        
     end
     'W m-2 um-1': begin 
        
        tellspec = mc_chfunits(wave,tellspec,0,1,0,IERROR=tellspec_error, $
                               OERROR=tellspec_error)
        scvega   = mc_chfunits(wave,scvega,0,1,0)

        state.nytitle = '!5f!D!7k!N!5 (W m!E-2!N !7l!5m!E-1!N'
        
     end
     'W m-2 Hz-1': begin

        tellspec = mc_chfunits(wave,tellspec,0,1,2, IERROR=tellspec_error, $
                               OERROR=tellspec_error)
        scvega   = mc_chfunits(wave,scvega,0,1,2)

        state.nytitle = '!5f!D!7m!N!5 (W m!E-2!N Hz!E-1!N' 
        
     end
     
     'Jy': begin
        
        tellspec = mc_chfunits(wave,tellspec,0,1,4,IERROR=tellspec_error, $
                               OERROR=tellspec_error)
        scvega   = mc_chfunits(wave,scvega,0,1,4)

        state.nytitle = '!5f!D!7m!N!5 (Jy' 
        
     end
     
  endcase
  
end
;
;===============================================================================
;
pro xtellcor_conkernel,state

  if state.continue lt 1 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.xtellcor_base)
     return
     
  endif

  case state.method of

     'Deconvolution': begin
     
        idx = where(*state.stdorders eq state.stdorder)
        minwave = min((*state.stdspec)[*,0,idx],MAX=maxwave,/NAN)
        
        xmc_conkern,(*state.stdspec)[*,0,idx],(*state.stdspec)[*,1,idx],$
                    *state.wvega,*state.fvega,*state.cfvega, $
                    *state.cf2vega,*state.awave,*state.atrans,wline, $
                    kernel,scale,vshift,maxdev,rmsdev, $
                    PARENT=state.xtellcor_base,XTITLE=state.xtitle,$
                    YTITLE=state.ytitle,CANCEL=cancel
        if cancel then return
        
;  Now we must scale the kernel to each order.
;  First, get this kernel in the instrument pixels.
        
        ndat = n_elements(kernel)
        kx_vegapix = findgen(ndat)
        result = gaussfit(kx_vegapix,kernel,a,NTERMS=3)
        
        z = where(*state.wvega gt minwave and *state.wvega lt maxwave)
        vdisp = mean(((*state.wvega)[z]-shift((*state.wvega)[z],1))[1:*])
        kx_spexpix = (kx_vegapix - a[1])*vdisp/total((*state.stddisp)[idx])
        
;  Now construct the kernels for the other orders.
             
        for i = 0, state.stdnorders-1 do begin

           minwave = min((*state.stdspec)[*,0,i],MAX=maxwave,/NAN)
           vdisp = mean(((*state.wvega)[z]-shift((*state.wvega)[z],1))[1:*])
           
           disp = total((*state.stddisp)[i])
           kx_spexwave = kx_spexpix*total((*state.stddisp)[i])
           
           del = max(kx_spexwave,MIN=min)-min
           npix = del/vdisp
           if not npix mod 2 then npix = npix + 1
           kx_vegawave = (findgen(npix)-npix/2.)*vdisp
           
           linterp,kx_spexwave,kernel,kx_vegawave,nkern
           
           nkern = nkern/total(nkern)
           key = 'Order'+string((*state.stdorders)[i],FORMAT='(i2.2)')
           str = (i eq 0) ? create_struct(key,nkern): $
                 create_struct(str,key,nkern)
           
        endfor
        
        state.vshift   = vshift
        state.scale    = scale
        *state.kernels = str
        state.maxdev   = maxdev
        state.rmsdev   = rmsdev
     
     end

     'IP': begin

;  Find the right IP based on the slit width

        z = where(state.slitw_arc eq state.ipcoeffs.slit,cnt)
        if cnt eq 1 then begin
           
;  Get the parameters
           
           parms = state.ipcoeffs[z].coeffs
           
;  Construct the kernels for each order
           
           for i = 0, state.stdnorders-1 do begin

              if state.fixeddisp then begin

;  Get the Vega dispersion
                 
                 min = min((*state.stdspec)[*,0,i],/NAN,MAX=max)
                 z = where(*state.wvega gt min and *state.wvega lt max)
                 vdisp = mean(((*state.wvega)[z]-$
                               shift((*state.wvega)[z],1))[1:*])
                 
                 vkernw_pix = (*state.stdfwhm)[i]/vdisp
                 
                 nkernel = round(7.*vkernw_pix)
                 if not nkernel mod 2 then nkernel = nkernel + 1
                 
                 kernel_vx = (findgen(nkernel)-fix(nkernel)/2)*vdisp
                 kernel_sx = kernel_vx/(*state.stddisp)[i]
                 
                 kern   = mc_instrprof(kernel_sx,parms,CANCEL=cancel)
                 if cancel then return
                 
                 key = 'Order'+string((*state.stdorders)[i],FORMAT='(i3.3)')
                 str1 = (i eq 0) ? create_struct(key,kern): $
                        create_struct(str1,key,kern)

              endif else begin

                 respower = mc_getrespower((*state.stdspec)[*,0,i], $
                                           state.slitw_pix,/FILLENDS,$
                                           DLAMBDA=dlambda,$
                                           DISPERSION=dispersion,CANCEL=cancel)
                 if cancel then return
                 
;  Find the minimum resolving power as this will be the widest kernel.
                 
                 min = min(respower,midx)

;  Get the Vega dispersion, std dispersion, and std fwhm
                 
                 tabinv,*state.wvega,(*state.stdspec)[midx,0,i],vidx
                 vdisp = (*state.wvega)[vidx+1]-(*state.wvega)[vidx]
                 stddisp = dispersion[midx]
                 stdfwhm = dlambda[midx]
                 
                 vkernw_pix = stdfwhm/vdisp
                 
                 nkernel = round(7.*vkernw_pix)
                 if not nkernel mod 2 then nkernel = nkernel + 1
                 
                 kernel_vx = (findgen(nkernel)-fix(nkernel)/2)*vdisp
                 kernel_sx = kernel_vx/(*state.stddisp)[i]
                 
                 kern   = mc_instrprof(kernel_sx,parms,CANCEL=cancel)
                 if cancel then return
                 
                 key = 'Order'+string((*state.stdorders)[i],FORMAT='(i3.3)')
                 str1 = (i eq 0) ? create_struct(key,kern): $
                        create_struct(str1,key,kern)

;  Now generate the scale array 

                 stdwmin = min((*state.stdspec)[*,0,i],MAX=stdwmax)
                 z = where(*state.wvega ge stdwmin and $
                           *state.wvega le stdwmax,nvega)

                 wvega = (*state.wvega)[z]
                 mask = make_array(n_elements(*state.wvega),VALUE=0)
                 mask = mask[z]
                 
                 nkernel = fltarr(nvega,/NOZERO)
                 for j = 0L,nvega-1L do begin
                    
                    vdisp = (*state.wvega)[z[j]+1]-(*state.wvega)[z[j]]
                    tabinv,(*state.stdspec)[*,0,i],(*state.wvega)[z[j]],sidx
                    stddisp = dispersion[round(sidx)]
                    stdfwhm = dlambda[round(sidx)]

                    vkernw_pix = stdfwhm/vdisp
                    
                    nkernel[j] = round(7D*vkernw_pix)
                    if not nkernel[j] mod 2 then nkernel[j] = nkernel[j] + 1
                                      
                 endfor

                 key = 'Order'+string((*state.stdorders)[i],FORMAT='(i3.3)')
                 str2 = (i eq 0) ? create_struct(key,[[wvega],[nkernel]]): $
                        create_struct(str1,key,[[wvega],[nkernel]])
                 
                                  
              endelse
                 
           endfor
           
           *state.kernels = str1
           if ~state.fixeddisp then *state.varrinfo = str2
           state.vshift   = 0.0
           state.scale    = 1.0
           
        endif else begin

           ok = dialog_message('No Kernel found.',/ERROR,$
                               DIALOG_PARENT=state.xtellcor_base)
           return           

        endelse
        
     end

  endcase

;  wvega = *state.wvega
;  fvega = *state.fvega
;  fcvega = *state.cfvega
;  fc2vega = *state.cf2vega
;  wstd = (*state.stdspec)[*,0,0]
;  fstd = (*state.stdspec)[*,1,0]
;  ustd = (*state.stdspec)[*,2,0]
;  kernel = str1
;  varrinfo = str2

;  save,wvega,fvega,fcvega,fc2vega,wstd,fstd,ustd,kernel,varrinfo, $
;       FILENAME='save.sav'
  
  state.continue = (state.scalelines eq 1) ? 2:3

end
;
;===============================================================================
;
pro xtellcor_contellspec,state

  if state.continue lt 3 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.xtellcor_base)
     return
     
  endif
  
  norders = fxpar(*state.objhdr,'NORDERS')
  match,*state.stdorders,*state.objorders,stdidx
  
  *state.tellspec = (*state.stdspec)[*,*,stdidx]
  *state.vegaspec = (*state.stdspec)[*,*,stdidx]
  (*state.vegaspec)[*,2,*] = 1.0

;  Get RVs if need be

  if ~state.scalelines eq 1 then begin

     rv = mc_cfld(state.rv_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     
     date = fix(strsplit(fxpar(*state.stdhdr,'AVE_DATE'),'-',/EXTRACT))
     time = fix(strsplit(fxpar(*state.stdhdr,'AVE_TIME'),':',/EXTRACT))
     ra   = float(strsplit(fxpar(*state.stdhdr,'RA'),':',/EXTRACT))
     dec  = float(strsplit(fxpar(*state.stdhdr,'DEC'),':',/EXTRACT))
     
     result = mc_earthvelocity(date[0],date[1],date[2],time[0],fix(ra[0]), $
                               fix(ra[1]),ra[2],fix(dec[0]),fix(dec[1]), $
                               dec[2],2000.0,/SILENT, CANCEL=cancel)
     if cancel then return     
     state.vshift = rv-result.vlsr

  endif
    
  print, 'Constructing Telluric Correction Spectra...'
  
  for i = 0, norders-1 do begin
     
     z = where(*state.stdorders eq (*state.objorders)[i],count)


     if state.scalelines then scales = (*state.scales)[*,z]
     
     if ~state.fixeddisp then varrinfo = *state.varrinfo 
     
     mc_mktellspec, (*state.stdspec)[*,0,z],(*state.stdspec)[*,1,z],$
                    (*state.stdspec)[*,2,z],state.vmag, $
                    (state.bmag-state.vmag),(*state.kernels).(i), $
                    *state.wvega,*state.fvega,*state.cfvega,*state.cf2vega, $
                    state.vshift,tellcor,tellcor_error,scvega, $
                    VARRINFO=varrinfo,SCALES=scales,CANCEL=cancel

;  Perform interpolations over regions defined in scalelines

     if size(*state.cutreg,/TYPE) eq 8 then begin
     
        cutreg = *state.cutreg
        ndat = n_elements(*cutreg.(i))
        
        if ndat ne 1 then begin
           
           nreg = ndat/2
           
           stdwave = (*state.stdspec)[*,0,z]
           stdflux = (*state.stdspec)[*,1,z]
           stderr  = (*state.stdspec)[*,2,z]
           
           nonan = mc_nantrim(stdwave,3)
           tstdwave = stdwave[nonan]
           tstdflux = stdflux[nonan]
           tstderr  = stderr[nonan]
           ttellcor = tellcor[nonan]
           ttellcor_error = tellcor_error[nonan]
           
           for j = 0, nreg-1 do begin
              
              xrange = reform((*cutreg.(i))[(j*2):(j*2+1)])
              tabinv,tstdwave,xrange,idx
              idx = round(idx)
              
              x = [tstdwave[idx[0]],tstdwave[idx[1]]]
              y = [ttellcor[idx[0]],ttellcor[idx[1]]]
              e = [ttellcor_error[idx[0]],ttellcor_error[idx[1]]]
              
              coeff  = mc_polyfit1d(x,y,1,/SILENT)
              coeffe = mc_polyfit1d(x,e,1,/SILENT)
              
              ttellcor[idx[0]:idx[1]]=poly(tstdwave[idx[0]:idx[1]],coeff)
              tellcor_error[idx[0]:idx[1]]=poly(tstdwave[idx[0]:idx[1]],coeffe)
              
           endfor
           tellcor[nonan] = ttellcor
           tellcor_error[nonan] = ttellcor_error
           
        endif

     endif
        
     xtellcor_changeunits,(*state.stdspec)[*,0,z],tellcor,tellcor_error, $
                          scvega,state
     (*state.tellspec)[*,1,i] = tellcor
     (*state.tellspec)[*,2,i] = tellcor_error
     (*state.vegaspec)[*,1,i] = scvega
     
  endfor
  
  state.continue = 4

end
;
;===============================================================================
;
pro xtellcor_getscales,state

  if state.continue lt 2 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.xtellcor_base)
     return
     
  endif

  vshift = state.vshift
  IP     = (state.method eq 'IP') ? 1:0
  
  wvega  = *state.wvega
  fvega  = *state.fvega
  fcvega = *state.cfvega
  
;  if state.stdobsmode eq 'ShortXD' then xputs = state.xputs
  
  match,*state.stdorders,*state.objorders,stdidx,objidx

  if ~state.fixeddisp then varrinfo = *state.varrinfo 

  xmc_scalelines,(*state.stdspec)[*,*,stdidx],(*state.stdorders)[stdidx],$
                 state.vmag,(state.bmag-state.vmag),wvega,fvega,fcvega, $
                 *state.cf2vega,*state.kernels,vshift,*state.objspec, $
                 *state.objorders,state.objnaps,*state.awave, $
                 *state.atrans,state.hlines,state.hnames,state.scale, $
                 scales,cutreg,PARENT=state.xtellcor_base, $
                 VARRINFO=varrinfo,XTITLE=state.xtitle, $
                 YTITLE=state.ytitle,CANCEL=cancel
  
  if not cancel then begin

     *state.scales = scales
     *state.cutreg = cutreg
     state.vshift  = vshift
     state.vrot    = 0.0

     state.continue = 3
     
  endif

end
;
;===============================================================================
;
pro xtellcor_loadspec,state

;  Get user inputs

  std = mc_cfld(state.std_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  stdfile = mc_cfld(state.stdspectra_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  stdfile = mc_cfile(state.stdpath+stdfile,WIDGET_ID=state.xtellcor_base, $
                 CANCEL=cancel)
  if cancel then return
  
  bmag = mc_cfld(state.bmag_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  vmag = mc_cfld(state.vmag_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  objfile = mc_cfld(state.objspectra_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  objfile = mc_cfile(state.objpath+objfile,WIDGET_ID=state.xtellcor_base, $
                 CANCEL=cancel)
  if cancel then return

;  Read std spectra 
  
  mc_readspec,stdfile,stdspec,stdhdr,stdobsmode,start,stop,stdnorders,stdnaps, $
              stdorders,stdxunits,stdyunits,slith_pix,slith_arc,slitw_pix, $
              slitw_arc,stdrp,stdairmass,xtitle,ytitle,instr,CANCEL=cancel
  if cancel then return

;  Read object spectra
  
  mc_readspec,objfile,objspec,objhdr,objobsmode,start,stop,objnorders,objnaps, $
              objorders,objxunits,objyunits,slith_pix,slith_arc,slitw_pix, $
              slitw_arc,objrp,objairmass,CANCEL=cancel
  if cancel then return  

;  Store necessary information

  state.std        = std
  *state.stdspec   = stdspec
  *state.stdorders = stdorders
  state.stdnorders = stdnorders
  *state.stdhdr    = stdhdr
  state.stdairmass = stdairmass
  state.stdobsmode = strtrim(stdobsmode,2)
  *state.stddisp   = mc_fxpar(stdhdr,'DISP*')
  *state.objspec   = objspec
  *state.objorders = objorders
  *state.objhdr    = objhdr
  state.objnaps    = objnaps
  state.bmag       = bmag
  state.vmag       = vmag
  state.xtitle     = xtitle
  state.ytitle     = ytitle
  state.slitw_pix  = slitw_pix
  state.slitw_arc  = slitw_arc
    
;  Generate arrays and initialize things

  state.dairmass = stdairmass-objairmass
  *state.shift   = fltarr(objnorders,objnaps)
  *state.stdfwhm = *state.stddisp*slitw_pix
  state.vshift   = 0.0  
  
;  Set the stdorder pulldown menus

  widget_control, state.stdorder_dl,SET_VALUE=string(stdorders,FORMAT='(I3)')
   
;  Determine xtellcor mode and modify widget accordingly

  readcol, filepath('xtellcor_modeinfo.dat',ROOT_DIR=state.packagepath,$
                    SUBDIR='data'),imode,vegarp,fixeddisp,method, $
           scalelines,output, $
           FORMAT='A,A,I,A,I,A',DELIMITER='|',COMMENT='#',/SILENT

;  Store the mode and whether you are running fixed dispersion or not
  
  z = where(strtrim(imode,2) eq state.stdobsmode,cnt)
  if cnt eq 0 then begin

     ok = dialog_message('Instrument mode unknown.',/ERROR,$
                         DIALOG_PARENT=state.xtellcor_base)
     return
    
  endif

  state.scalelines = scalelines[z]
  state.fixeddisp = fixeddisp[z]
  
;  Determine kernel generation method and modify widget accordingly
  
  meth = fix(strsplit(method[z],' ',/EXTRACT))
  state.method = (meth[0] eq 0) ? 'Deconvolution':'IP'
  
  widget_control, state.method_bg,SET_VALUE=meth[0]
  widget_control, state.method_bg,SENSITIVE=meth[1]
  widget_control, state.stdorder_dl,SENSITIVE=~meth[0]
  
  if n_elements(meth) eq 3 then begin

     z = where(stdorders eq meth[2],cnt)
     if cnt ne 0 then begin

        widget_control, state.stdorder_dl,SET_DROPLIST_SELECT=z
        state.stdorder   = stdorders[z]
        
     endif else state.stdorder = stdorders[0]

  endif

  sensitive=(state.method eq 'IP') ? 0:1
  widget_control, state.stdorder_dl,SENSITIVE=sensitive

;  Set the scale line box sensitivities

  widget_control, state.scalelines_but,SENSITIVE=state.scalelines
  widget_control, state.rv_fld[0],SENSITIVE=~state.scalelines
  
;  Load the Vega model

  restore, filepath('vega'+strtrim(vegarp[z],2)+'.sav', $
                    ROOT_DIR=state.spextoolpath,SUBDIR='data')
    
  *state.wvega = wvin
  *state.cf2vega = fc2vin
  *state.cfvega = fcvin
  *state.fvega = fvin  

;  Set the output types

  out = fix(strsplit(output[z],' ',/EXTRACT))
  widget_control, state.output_bg,SET_VALUE=out

  state.telluricoutput=out[0]
  state.vegaoutput=out[1]
  
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

;  Compute airmass difference between obj and std.

  color = (abs(stdairmass-objairmass) gt 0.1) ? 17:16
  
  wset, state.message
  erase,COLOR=color
  xyouts,10,8,'Std Airmass:'+string(stdairmass,FORMAT='(f7.4)')+ $
         ', Obj Airmass:'+string(objairmass,FORMAT='(f7.4)')+ $
         ', (Std-Obj) Airmass: '+ $
         string((stdairmass-objairmass),FORMAT='(f7.4)'),/DEVICE,$
         CHARSIZE=1.1,FONT=0


  
  state.continue=1
  
end
;
;===============================================================================
;
pro xtellcor_getshifts,state

  if state.continue lt 4 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.xtellcor_base)
     return
     
  endif

  shifts = xmc_findshifts(*state.objspec,*state.tellspec,*state.objorders, $
                          *state.awave,*state.atrans,state.xtitle,CANCEL=cancel)
  if ~cancel then begin

     *state.shift = shifts
     state.continue=5
     
  endif
  
end
;
;===============================================================================
;
pro xtellcor_writefile,state

  if state.continue lt 4 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.xtellcor_base)
     return
     
  endif
  
  obj = mc_cfld(state.objoname_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  stdfile = mc_cfld(state.stdspectra_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
;
;  Write the telluric correction spectrum to disk
;
  
  if state.telluricoutput then begin
     
;  Get info from obj and std hdr.
     
     orders    = fxpar(*state.objhdr,'ORDERS',COMMENT=corders)
     norders   = fxpar(*state.objhdr,'NORDERS',COMMENT=cnorders)
     obsmode   = fxpar(*state.stdhdr,'MODE',COMMENT=cobsmode)
     xunits    = fxpar(*state.stdhdr,'XUNITS',COMMENT=cxunits)
     xtitle    = fxpar(*state.stdhdr,'XTITLE',COMMENT=cxtitle,COUNT=count)
     
;  For backwards compatibility
     
     if count eq 0 then begin
        
        xtitle  = '!7k!5 ('+strtrim(xunits,2)+')'
        cxtitle = 'IDL X Title'
        
     endif
     
;  Create hdr for the telluric correction spectrum
     
     fxhmake,hdr,*state.tellspec
     fxaddpar,hdr,'FILENAME',strtrim(obj+'_tellspec.fits',2)
     fxaddpar,hdr,'CREPROG','xtellcor', ' Creation IDL program'  
     fxaddpar,hdr,'ORDERS',orders,corders
     fxaddpar,hdr,'NORDERS',norders,cnorders
     fxaddpar,hdr,'A0VStd',state.std,' Telluric correction A0 V standard'
     fxaddpar,hdr,'A0VBmag',state.bmag,' B-band magnitude'
     fxaddpar,hdr,'A0VVmag',state.vmag,' V-band magnitude'
     fxaddpar,hdr,'NAPS',1, ' Number of apertures'
     fxaddpar,hdr,'AM',state.stdairmass, 'Average airmass'

     fxaddpar,hdr,'DELTAAM',state.dairmass,' Average of (std-obj) airmass'
     fxaddpar,hdr,'TELMETH',strtrim(state.method,2), $
              ' Telluric correction method'
     fxaddpar,hdr,'VEGADV',state.vshift, ' Vega velocity shift in km s-1'
     
     
     history = 'These telluric correction spectra were constructed from the '+$
               'spectra '+strtrim(stdfile,2)+'.  The velocity shift of' + $
               ' Vega is '+strtrim(state.vshift,2)+' (km s-1).  '
     
     if state.method eq 'Deconvolution' then begin

        fxaddpar,hdr,'TCMaxDev',state.maxdev*100, $
                 ' The maxmimum % deviation of Vega-data'
        
        fxaddpar,hdr,'TCRMSDev',state.rmsdev*100, $
                 ' The RMS % deviation of Vega-model'
        
        history = history+'The Vega model was modified using the ' + $
                  'deconvolution method.  The maximum deviation and ' + $
                  'rms deviation between the modified Vega model and the ' + $
                  'data over the kernel wavelength range are '+ $
                  strtrim(state.maxdev*100.0,2)+'% and '+ $
                  strtrim(state.rmsdev*100.0,2)+'%, respectively.'
        
     endif else begin
        
        history = history+' The Vega model was modified using the IP method.'
        
     endelse

     fxaddpar,hdr,'XUNITS',xunits,cxunits
     fxaddpar,hdr,'XTITLE',xtitle,cxtitle
     fxaddpar,hdr,'YUNITS',strcompress(state.units,/RE)+'/DNs-1', $
              ' Units of the Y axis'
     fxaddpar,hdr,'YTITLE',state.nytitle+' / DN s!U-1!N)','IDL Y title'
     
     history = mc_splittext(history,68,CANCEL=cancel)
     if cancel then return
     sxaddhist,history,hdr
     
;  Write it  to disk
     
     writefits,state.objpath+obj+'_tellspec.fits',*state.tellspec,hdr

     print
     print,'Wrote the telluric correction spectrum to '+ $
           strtrim(state.objpath+obj,2)+'_tellspec.fits'
     print

  endif  
;
;  Write the convolved resampled Vega spectrum to disk
;
  
  if state.vegaoutput then begin

  rshift = 1.0 + (state.vshift/2.99792458D5)
  (*state.vegaspec)[*,0,*] = (*state.vegaspec)[*,0,*]*rshift
     
;  Get info from std hdr.
    
     orders  = fxpar(*state.objhdr,'ORDERS',COMMENT=corders)
     norders = fxpar(*state.objhdr,'NORDERS',COMMENT=cnorders)
     mode    = fxpar(*state.stdhdr,'MODE',COMMENT=cmode)
     xunits  = fxpar(*state.stdhdr,'XUNITS',COMMENT=cxunits)
     xtitle  = fxpar(*state.stdhdr,'XTITLE',COMMENT=cxtitle)
     
;  Create FITS header for the Vega spectrum
     
     delvarx,hdr
     fxhmake,hdr,*state.vegaspec
     
     sxaddpar,hdr,'FILENAME',strtrim(obj+'_modvega.fits',2)
     sxaddpar,hdr,'CREPROG','xtellcor', ' Creation IDL program'  
     fxaddpar,hdr,'ORDERS',orders,corders
     sxaddpar,hdr,'NORDERS',norders,cnorders
     sxaddpar,hdr,'OBJECT','Convolved Vega Spectrum'
     sxaddpar,hdr,'MODE',mode,mode 
     fxaddpar,hdr,'NAPS',1, ' Number of apertures'
          
     history = 'This Vega spectrum has been scaled to a V magnitude of '+$
               strtrim(state.vmag,2)+', shifted by '+ $
               strtrim(state.vshift,2)+$
               ' km s-1, convolved with the kernel, and resampled '+$
               'onto the wavelength grid of '+strtrim(stdfile,2)+'.'
     
     if state.method eq 'Deconvolution' then begin
        
        fxaddpar,hdr,'TCMaxDev',state.maxdev*100, $
                 ' The maxmimum % deviation of Vega-data'
        
        fxaddpar,hdr,'TCRMSDev',state.rmsdev*100, $
                 ' The RMS % deviation of Vega-data'
        
        history = history+'  The Vega model was modified using the ' + $
                  'deconvolution method.  The maximum deviation and ' + $
                  'rms deviation between the modified Vega model and the ' + $
                  'data over the kernel wavelength range are '+ $
                  strtrim(state.maxdev*100.0,2)+'% and '+ $
                  strtrim(state.rmsdev*100.0,2)+'%, respectively.'
        
     endif else begin
        
        history = history+' The Vega model was modified using the IP method.'
        
     endelse

     fxaddpar,hdr,'XUNITS',xunits,cxunits
     fxaddpar,hdr,'XTITLE',xtitle,cxtitle
     fxaddpar,hdr,'YUNITS',strcompress(state.units,/RE), 'Units of the Y axis'
     fxaddpar,hdr,'YTITLE',state.nytitle+')','IDL Y title'

     history = mc_splittext(history,68,CANCEL=cancel)
     if cancel then return
     sxaddhist,history,hdr
     
     writefits,state.objpath+obj+'_modVega.fits',*state.vegaspec,hdr
     
     print
     print,'Wrote the Vega spectrum to '+strtrim(state.objpath+obj,2)+ $
           '_vega.fits'
     print

  endif
  
;  Write the telluric corrected object spectrum to disk.  First
;  telluric correct the object spectra
  
  corspec  = *state.objspec
  for i = 0, n_elements(*state.objorders)-1 do begin
     
     z = where((*state.stdorders) eq (*state.objorders)[i],count)
     if count eq 0 then goto, cont
     
     for j = 0, state.objnaps-1 do begin
        
        k = i*state.objnaps+j
        
;  Interpolate telluric spectrum onto object wavelength sampling.
        
        mc_interpspec,(*state.tellspec)[*,0,i],(*state.tellspec)[*,1,i],$
                   (*state.objspec)[*,0,k],nflux,nerror,$
                   IYERROR=(*state.tellspec)[*,2,i],CANCEL=cancel
        if cancel then return

;  Interpolate mask into object wavelength sampling

        mc_interpflagspec,(*state.tellspec)[*,0,i],$
                          byte((*state.tellspec)[*,3,i]),$
                          (*state.objspec)[*,0,k],nflag,CANCEL=cancel
        if cancel then return

        
;  Now shift spectrum.
        
        x = findgen(n_elements((*state.objspec)[*,0,k]))
        mc_interpspec,x+(*state.shift)[i,j],nflux,x,nnflux,nnerror, $
                      IYERROR=nerror,CANCEL=cancel
        if cancel then return

        mc_interpflagspec,x+(*state.shift)[i,j],byte(nflag),x,nnflag,$
                          CANCEL=cancel
        if cancel then return

        corspec[*,1,k] = nnflux*(*state.objspec)[*,1,k]

        corspec[*,2,k] = sqrt( nnflux^2 * (*state.objspec)[*,2,k]^2 + $
                               (*state.objspec)[*,1,k]^2 * nnerror^2 )
        
        result = mc_combflagstack(byte([[[(*state.objspec)[*,3,k]]], $
                                   [[nnflag]]]),CANCEL=cancel)
        if cancel then return
        
        corspec[*,3,k] = result
        
     endfor
     
     cont:
     
  endfor
  
;  Now write it to disk
  
  hdr = *state.objhdr

  hdrinfo = mc_gethdrinfo(*state.objhdr,state.keywords,/IGNOREMISSING,$
                          CANCEL=cancel)
  if cancel then return

  fxhmake,newhdr,corspec
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

;  Update keywords
  
  sxaddpar,newhdr,'FILENAME',strtrim(obj+'.fits',2)
  sxaddpar,newhdr,'CREPROG','xtellcor', ' Creation program'  
  sxaddpar,newhdr,'YUNITS',strcompress(state.units,/RE), ' Units of the Y axis'
  fxaddpar,newhdr,'YTITLE',state.nytitle+')',' IDL Y title'
  
;  Add xtellcor keywords

  if state.userkeywords[0] ne '' then before = state.userkeywords[0]

  sxaddpar,newhdr,'A0VStd',state.std, $
           ' Telluric Correction A0 V Standard',BEFORE=before
  sxaddpar,newhdr,'A0VBmag',state.bmag,' B-band magnitude',BEFORE=before
  sxaddpar,newhdr,'A0VVmag',state.vmag,' V-band magnitude',BEFORE=before
  sxaddpar,newhdr,'DELTAAM',state.dairmass, $
           ' Average of (std-obj) airmass',BEFORE=before
  sxaddpar,newhdr,'TELMETH',strtrim(state.method,2), $
           ' Telluric correction method',BEFORE=before
  sxaddpar,newhdr,'VEGADV',state.vshift, ' Vega velocity shift (km s-1)', $
           BEFORE=before

  if state.method eq 'Deconvolution' then begin
     
     sxaddpar,newhdr,'TCMaxDev',state.maxdev*100, $
              ' The maxmimum % deviation of Vega-data',BEFORE=before
     
     sxaddpar,newhdr,'TCRMSDev',state.rmsdev*100, $
              ' The RMS % deviation of Vega-data',BEFORE=before
  
  endif
  sxaddpar,newhdr,'Telfile',obj+'_tellspec.fits', $
           ' The telluric correction file',BEFORE=before
 
;  Write new xtellcor history
  
  
  history = 'The Vega model was shifted by '+strtrim(state.vshift,2)+ $
            ' km s-1.  '
  
  if state.method eq 'Deconvolution' then begin

     history = history+'The maximum deviation and rms deviation between ' + $
               'the modified Vega model and the data over the kernel ' + $
               'wavelength range are '+strtrim(state.maxdev*100.0,2)+'% and '+$
               strtrim(state.rmsdev*100.0,2)+'%, respectively.  '

  endif

  if total(*state.shift) ne 0 then begin
    
     for i = 0, state.objnaps-1 do begin
        
        history = history+'The telluric correction spectra for aperture '+ $
                  string(i+1,FORMAT='(i2.2)')+' were shifted by '+ $
                  strjoin(string((*state.shift)[*,i],FORMAT='(f+5.2)'),', ')+ $
                  ' pixels.  '
        
     endfor

  endif
    
  sxaddhist,' ',newhdr
  sxaddhist,'######################## Xtellcor History ' + $
            '#########################',newhdr
  sxaddhist,' ',newhdr

  history = mc_splittext(history,67,CANCEL=cancel)
  if cancel then return
  sxaddhist,history,newhdr

;  Write to disk
  
  writefits,state.objpath+obj+'.fits',corspec,newhdr
  xvspec,state.objpath+obj+'.fits',/PLOTLINMAX
  
  print
  print,'Wrote the corrected spectrum to '+strtrim(state.objpath+obj,2)+'.fits'
  print

end
;
;===============================================================================
;
; ------------------------------Main Program---------------------------------- 
;
;===============================================================================
;
pro xtellcor,instrument,FINISH=finish,BASIC=basic,GENERAL=general

;  Check for other xtellcor calls
  
  if keyword_set(FINISH) then begin
     
     xtellcor_finish
     return
     
  endif

  if keyword_set(BASIC) then begin
     
     xtellcor_basic
     return
     
  endif

  if keyword_set(GENERAL) then begin
     
     xtellcor_general
     return
     
  endif
  
;  Get spextool and instrument information 
  
  mc_getspextoolinfo,spextoolpath,packagepath,spextool_keywords,instrinfo, $
                     notirtf,version,INSTRUMENT=instrument,CANCEL=cancel
  if cancel then return
  
;  Get hydrogen lines
  
  readcol,filepath('HI.dat',ROOT_DIR=spextoolpath,SUBDIR='data'),$
          hlines,hnames,FORMAT='D,A',COMMENT='#',DELIMITER='|',/SILENT
  
;  Get IP profile coefficients

  readcol,filepath('IP_coefficients.dat',ROOT_DIR=packagepath,SUBDIR='data'),$
          slit,p0,p1,p2,FORMAT='F,D,D',COMMENT='#',/SILENT
  
  nslits = n_elements(slit)
  
  ipcoeffs = {slit:slit[0],coeffs:[double(p0[0]),double(p1[0]),double(p2[0])]}
  if nslits gt 1 then begin
     
     ipcoeffs = replicate(ipcoeffs,nslits)
     for i = 1,nslits-1 do begin
        
        ipcoeffs[i].slit = slit[i]
        ipcoeffs[i].coeffs = [double(p0[i]),double(p1[i]),double(p2[i])]
        
     endfor
     
  endif
  
;  Load color table

  mc_mkct
  
;  Set the fonts

  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return

;  Build three structures which will hold important info.
         
  state = {airmass:'',$
           awave:ptr_new(2),$
           atrans:ptr_new(2),$
           bmag:0.,$
           bmag_fld:[0,0],$
           cfvega:ptr_new(fltarr(2)),$
           cf2vega:ptr_new(fltarr(2)),$
           continue:0L,$
           cutreg:ptr_new(2),$
           dairmass:0.0,$
           fixeddisp:0,$
           fvega:ptr_new(fltarr(2)),$
           fwhm_fld:[0L,0L],$
           hlines:hlines,$
           hnames:hnames,$
           instrument:instrinfo.instrument,$
           ipcoeffs:ipcoeffs,$
           kernels:ptr_new(fltarr(2)),$
           keywords:[spextool_keywords,instrinfo.xtellcor_keywords,'HISTORY'],$
           maxdev:0.0,$
           message:0L,$
           method:'IP',$
           method_bg:0L,$
           nytitle:'',$
           nstd:ptr_new(fltarr(2)),$
           objhdr:ptr_new(fltarr(2)),$
           objorders:ptr_new(fltarr(2)),$
           objnaps:0,$
           objpath:'',$
           objoname_fld:[0L,0L],$
           objspec:ptr_new(fltarr(2)),$
           objspectra_fld:[0,0],$
           output_bg:0L,$
           packagepath:packagepath,$
           rmsdev:0.0,$
           rv_fld:[0L,0L],$
           scale:0.,$
           scales:ptr_new(fltarr(2)),$
           scalelines:1,$
           scalelines_but:0L,$
           shift:ptr_new(fltarr(2)),$
           shiftobjap:0,$
           slitw_arc:0.,$
           slitw_pix:0.,$
           spextoolpath:spextoolpath,$
           std:'',$
           std_fld:[0L,0L],$
           stdairmass:0.,$
           stdhdr:ptr_new(fltarr(2)),$
           stddisp:ptr_new(fltarr(2)),$
           stdfwhm:ptr_new(fltarr(2)),$
           stdnorders:0,$
           stdobsmode:'',$
           stdorders:ptr_new(fltarr(2)),$
           stdorder:0,$
           stdorder_dl:0L,$
           stdpath:'',$
           stdspec:ptr_new(fltarr(2)),$
           stdspectra_fld:[0L,0L],$
           telluricoutput:1,$
           tellspec:ptr_new(fltarr(2)),$
           units:'ergs s-1 cm-2 A-1',$
           userkeywords:instrinfo.xtellcor_keywords,$
           varrinfo:ptr_new(2),$
           vegaoutput:0,$
           vegaspec:ptr_new(fltarr(2)),$
           vmag:0.,$
           vmag_fld:[0,0],$
           vrot:0.0,$
           vshift:0.,$
           wline:0.,$
           wvega:ptr_new(fltarr(2)),$
           xtellcor_base:0L,$
           xtitle:'',$
           ytitle:''}

  title = 'Xtellcor '+version+' for '+state.instrument
  
  state.xtellcor_base = widget_base(TITLE=title,$
                                      EVENT_PRO='xtellcor_event',$
                                      /COLUMN)
  
     button = widget_button(state.xtellcor_base,$
                            FONT=buttonfont,$
                            VALUE='Done',$
                            UVALUE='Quit')
     
     message = widget_draw(state.xtellcor_base,$
                           FRAME=2,$
                           XSIZE=1,$
                           YSIZE=25)
     
     row_base = widget_base(state.xtellcor_base,$
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

              fld = coyote_field2(box1_base,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE='A0 V Standard:',$
                                  UVALUE='Standard Spectra Field',$
                                  XSIZE=18,$
;                                  VALUE='HD asdf',$
                                  /CR_ONLY,$
                                  TEXTID=textid)
              state.std_fld = [fld,textid]
              
              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 button = widget_button(row,$
                                        FONT=buttonfont,$
                                        VALUE='Std Spectra',$
                                        UVALUE='Standard Spectra Button')
                 
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE=':',$
                                     UVALUE='Standard Spectra Field',$
                                     XSIZE=18,$
;                                     VALUE='cspectra176-183.fits',$
                                     /CR_ONLY,$
                                     TEXTID=textid)
                 state.stdspectra_fld = [fld,textid]
                 
              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE='Std Mag (B,V):',$
                                     UVALUE='B Magnitude Field',$
                                     XSIZE=6,$
                                     VALUE='7',$
                                     /CR_ONLY,$
                                     TEXTID=textid)
                 state.bmag_fld = [fld,textid]
                 
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     UVALUE='V Magnitude Field',$
                                     XSIZE=6,$
                                     VALUE='7',$
                                     TITLE='',$
                                     /CR_ONLY,$
                                     TEXTID=textid)
                 state.vmag_fld = [fld,textid]

              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 buton = widget_button(row,$
                                       FONT=buttonfont,$
                                       VALUE='Obj Spectra',$
                                       UVALUE='Object Spectra Button')
                 
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE=':',$
                                     UVALUE='Object Spectra Field',$
;                                     VALUE='cspectra176-183.fits',$
                                     XSIZE=18,$
                                     /CR_ONLY,$
                                     TEXTID=textid)
                 state.objspectra_fld = [fld,textid]
                 
              load = widget_button(box1_base,$
                                   VALUE='Load Spectra',$
                                   FONT=buttonfont,$
                                   UVALUE='Load Spectra')
              
           box_base = widget_base(col1_base,$
                                  /COLUMN,$
                                  FRAME=2)
           
              label = widget_label(box_base,$
                                   VALUE='2.  Construct Convolution Kernel',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)

              state.method_bg = cw_bgroup(box_base,$
                                          FONT=buttonfont,$
                                          ['Deconvolution','IP'],$
                                          /ROW,$
                                          /RETURN_NAME,$
                                          /NO_RELEASE,$
                                          /EXCLUSIVE,$
                                          LABEL_LEFT='Method:',$
                                          UVALUE='Method',$
                                          SET_VALUE=value)
              
              state.stdorder_dl = widget_droplist(box_base,$
                                                  FONT=buttonfont,$
                                                  TITLE='Order:',$
                                                  VALUE='1',$
                                                  UVALUE='Standard Order')
              
              button = widget_button(box_base,$
                                     VALUE='Construct Kernel',$
                                     UVALUE='Construct Kernel',$
                                     FONT=buttonfont)

        col2_base = widget_base(row_base,$
                                /COLUMN)

           box_base = widget_base(col2_base,$
                                  /COLUMN,$
                                  FRAME=2)

              label = widget_label(box_base,$
                       VALUE='3.  Construct Telluric Spectra',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)
              
              fld = coyote_field2(box_base,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE='Std Vrad (km s-1):',$
                                  UVALUE='Radial velocity',$
                                  XSIZE=7,$
                                  VALUE='0.0',$
                                  /CR_ONLY,$
                                  TEXTID=textid)
              state.rv_fld = [fld,textid]

              state.scalelines_but = widget_button(box_base,$
                                                   VALUE='Scale Lines',$
                                                   UVALUE='Scale Lines',$
                                                   FONT=buttonfont)
              
              value =['ergs s-1 cm-2 A-1','ergs s-1 cm-2 Hz-1',$
                      'W m-2 um-1','W m-2 Hz-1','Jy']
              units_dl = widget_droplist(box_base,$
                                         FONT=buttonfont,$
                                         TITLE='Units:',$
                                         VALUE=value,$
                                         UVALUE='Spectrum Units')
              
              constructspec = widget_button(box_base,$
                                           VALUE='Construct Telluric Spectra',$
                                           UVALUE='Construct Telluric Spectra',$
                                            FONT=buttonfont)
              
           box_base = widget_base(col2_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box_base,$
                                   VALUE='4.  Determine Shifts',$
                                   FONT=buttonfont)
                          
              shift = widget_button(box_base,$
                                    VALUE='Find Shifts',$
                                    UVALUE='Get Shifts',$
                                    FONT=buttonfont)


           box_base = widget_base(col2_base,$
                                  /COLUMN,$
                                  FRAME=2)
           
              label = widget_label(box_base,$
                                   VALUE='5.  Write File',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)
              
              oname = coyote_field2(box_base,$
                                    LABELFONT=buttonfont,$
                                    FIELDFONT=textfont,$
                                    TITLE='File Name:',$
                                    UVALUE='Output File Oname',$
                                    xsize=18,$
                                    textID=textid)
              state.objoname_fld = [oname,textid]
              
              state.output_bg = cw_bgroup(box_base,$
                                          FONT=buttonfont,$
                                          ['Telluric','Model'],$
                                          /ROW,$
                                          /RETURN_NAME,$
                                          /NONEXCLUSIVE,$
                                          LABEL_LEFT='Additional:',$
                                          UVALUE='Additional Output')
              
              write = widget_button(box_base,$
                                    VALUE='Write File',$
                                    UVALUE='Write File',$
                                    FONT=buttonfont)
              
  help = widget_button(state.xtellcor_base,$
                       VALUE='Help',$
                       UVALUE='Help',$
                       FONT=buttonfont)
  
; Get things running.  Center the widget using the Fanning routine.
            
  cgcentertlb,state.xtellcor_base
  widget_control, state.xtellcor_base, /REALIZE
  widget_control, message, GET_VALUE=x
  state.message = x
  
 widget_geom = widget_info(state.xtellcor_base, /GEOMETRY)
 widget_control, message, XSIZE=widget_geom.xsize-17
 erase, color=196
 
; Start the Event Loop. This will be a non-blocking program.

  XManager, 'xtellcor', $
            state.xtellcor_base, $
            CLEANUP='xtellcor_cleanup',$
            /NO_BLOCK
  
  widget_control, state.xtellcor_base, SET_UVALUE=state, /NO_COPY

end
