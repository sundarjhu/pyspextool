;+
; NAME:
;     xvspec
;
; PURPOSE:
;     Displays Spextool spectral FITS data file.
;    
; CATEGORY:
;     Widget
;
; CALLING SEQUENCE:
;     xvspec,[afile],[bfile],POSITION=position,LADDER=ladder,$
;            CONTINUOUS=continuous,MODE=mode,PLOTWINSIZE=plotwinsize,$
;            PLOTLINMAX=plotlinmax,PLOTREPLACE=plotreplace,PLOTFIX=plotfix,$
;            PLOTOPTFAIL=plotoptfail,PLOTATMOSPHERE=plotatmosphere,$
;            NOUPDATE=noupdate,GROUP_LEADER=group_leader,CANCEL=cancel
;     
; INPUTS:
;     None
;    
; OPTIONAL INPUTS:
;     afile - A string giving the filename of a SpeX spectral FITS
;             image
;     bfile - A string giving the filename of a SpeX spectral FITS
;             image.  The file must effectively be identical to the
;             afile. That is, the bfile can be the "B Beam" spectrum
;             of a pair subtracted extraction.  
;
; KEYWORD PARAMETERS:
;     POSITION      - A 2-element array giving the position of the
;                     widget on the screen in normalized coordinates.
;     LADDER        - Set to plot as a ladder
;     CONTINUOUS    - Set to plot as a continuous spectrum
;     MODE          - 'Ladder' or 'Continuous'.  Defaults to Ladder.
;                     The LADDER and CONTINUOUS keywords override this keyword.
;     PLOTWINSIZE   - A 2-element array giving the plot window size.
;     PLOTLINMAX    - Set to plot pixels beyond linearity maximum.
;     PLOTREPLACE   - Set to plot pixels that have been replaced.
;     PLOTFIX       - Set to plot pixels that have been fixed.
;     PLOTOPTFAIL   - Set to plot pixels that have failed optimal
;                     extraction.
;     PLOTAMOSPHERE - Set to plot pixels the atmospheric transmission.
;     NOUPDATE      - Set to ignore a PLOTWINSIZE,LADDER, CONTINUOUS, MODE
;                     keywords.
;     GROUP_LEADER  - The widget ID of an existing widget that serves
;                     as "group leader" for the newly-created widget. 
;     CANCEL        - Set on return if there is a problem
;     
; OUTPUTS:
;     None
;     
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     xvspec_state
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     Easy
;
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;     M. Cushing, Institute for Astronomy, University of Hawaii
;-
;
;===============================================================================
;
; ----------------------------Support procedures------------------------------ 
;
;===============================================================================
;
pro xvspec_initcommon

  common xvspec_state,state

  mc_mkct
  device, RETAIN=2
  
;  Get fonts
  
  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return
  
;  get Spextool path.
  
  spextoolpath = file_dirname(file_dirname( $
                 file_which('spextool_instrument.dat'),/MARK))

;  Get the H lines
  
  readcol,filepath('HI.dat',ROOT_DIR=spextoolpath,SUBDIR='data'),$
          hlines,hnames,FORMAT='D,A',COMMENT='#',DELIMITER='|',/SILENT  

  
;  Get screen size
  
  screensize = get_screen_size()
  
  state = {absflxwrange:[0.,0.],$
           absflxyrange:[0.,0.],$
           abssnrwrange:[0.,0.],$
           abssnryrange:[0.,0.],$
           absuncwrange:[0.,0.],$
           absuncyrange:[0.,0.],$
           afile:'',$
           ahdr:ptr_new(strarr(2)),$
           altcolor:2,$
           ap:0,$
           amspectra:ptr_new(fltarr(2)),$                ; modified spectra
           aspectra:ptr_new(fltarr(2)),$                 ; raw spectra
           atrans:ptr_new(2),$
           awave:ptr_new(2),$
           bfile:'',$
           bhdr:ptr_new(strarr(2)),$
           bspectra:ptr_new(fltarr(2)),$
           bmspectra:ptr_new(fltarr(2)),$
           buffer:0,$
           buttonfont:buttonfont,$
           but_aperture:0L,$
           but_ap:ptr_new(0),$
           but_buffer:0L,$
           but_buffera:0L,$
           but_bufferb:0L,$
           but_continuous:0L,$
           but_fixyrange:0L,$
           but_flags:[0L,0L,0L,0L],$
           but_flx:0L,$
           but_flux:0L,$
           but_fluxu:[0L,0L,0L,0L,0L,0L,0L],$
           but_help:0L,$
           but_ladder:0L,$
           but_order:0L,$
           but_ord:ptr_new(0),$
           but_plotatmos:0L,$
           but_plothlines:0L,$
           but_plotuserlines:0L,$           
           but_smooth:0L,$
           but_snr:0L,$
           but_snrcut:0L,$
           but_unc:0L,$
           but_wavelength:0L,$
           but_waveu:[0L,0L,0L],$
           but_writefits:0L,$
           but_writeascii:0L,$
           but_xlog:0L,$
           but_xranges:0L,$
           but_ylog:0L,$
           but_2coloraltbut:0L,$
           but_3coloraltbut:0L,$
           charsize:1.5,$
           color:3,$
           cursormode:'None',$
           filenamepanel:0,$
           flxyranges:ptr_new(fltarr(2)),$
           flxwrange:[0.,0.],$
           flxyrange:[0.,0.],$
           funits:ptr_new(2),$
           fwhm:0.,$
           hlines:hlines,$
           hnames:hnames,$
           mbut_2color:0L,$
           mbut_3color:0L,$
           mbut_plotatmos:0L,$
           mbut_buffera:0L,$
           mbut_bufferb:0L,$
           mbut_ladder:0L,$
           mbut_continuous:0L,$
           mbut_flx:0L,$
           mbut_snr:0L,$
           mbut_unc:0L,$
           message:0L,$
           mode:'Ladder',$
           norders:0,$
           naps:0,$
           nbuffers:0,$
           orders:ptr_new(2),$
           panel:0L,$
           path:'',$
           pfunits:['W m-2 um-1',$
                    'ergs s-1 cm-2 A-1',$
                    'W m-2 Hz-1',$
                    'ergs s-1 cm-2 Hz-1',$
                    'Jy',$
                    'W m-2',$
                    'ergs s-1 cm-2'],$
           pixmap_wid:0L,$
           pixperorder:250,$
           plotatmosphere:0L,$
           plotbase:0L,$
           plotreplacepixel:1L,$
           plotfixpixel:1L,$
           plothlines:0,$
           plotoptfail:1L,$
           plotsatpixel:1L,$
           plotuserlines:0L,$             
           plotwin:0L,$
           plotwin_wid:0L,$
           plotxranges:0L,$
           plot_size:screensize*[0.5,0.5],$
           pscale:!p,$
           pfile:'',$
           pbuffer:'',$
           pspectra:ptr_new(2),$
           pytitle:'',$
           pyranges:ptr_new(2),$
           pabsyrange:[0.,0.],$
           pabswrange:[0.,0.],$
           pyrange:[0.,0.],$
           pwrange:[0.,0.],$
           pwunits:['um','nm','A'],$
           ranges:0L,$
           reg:[[!values.d_nan,!values.d_nan],$
                [!values.d_nan,!values.d_nan]],$
           scrollbars:0,$
           scroll_size:screensize*[0.5,0.5],$
           slider:0L,$
           sliderval:50,$
           smoothingpanel:0,$
           snryranges:ptr_new(fltarr(2)),$
           snrwrange:[0.,0.],$
           snryrange:[0.,0.],$
           snrcutpanel:0,$
           snrcut:0,$
           spectype:'Flux',$
           spextoolpath:spextoolpath,$
           textfont:textfont,$
           tmp_fld:[0L,0L],$
           uncyranges:ptr_new(fltarr(2)),$
           uncwrange:[0.,0.],$
           uncyrange:[0.,0.],$
           userlines:ptr_new(!values.f_nan),$
           usernames:ptr_new(''),$
           winbuffer:[0L,0L],$
           wranges:ptr_new(fltarr(2)),$
           wunits:ptr_new(2),$
           xlog:0,$
           xmax_fld:[0L,0L],$
           xmin_fld:[0L,0L],$				
           xranges:ptr_new(2),$
           xscale:!x,$
           xtitle:'',$
           xunits:'',$
           xvspec_base:0L,$
           ylog:0,$
           ymax_fld:[0L,0L],$
           ymin_fld:[0L,0L],$
           yscale:!y,$
           ytitle:['','',''],$
           yunits:''}


end
;
;===============================================================================
;
pro xvspec_addpanel,SMOOTH=smooth,ASCII=ascii,FITS=fits,SNRCUT=snrcut

  common xvspec_state

  if keyword_set(SNRCUT) then begin

     widget_control, state.but_snrcut,/SET_BUTTON
     
     state.panel = widget_base(state.xvspec_base,$
                               /BASE_ALIGN_CENTER,$
                               FRAME=2,$
                               /ROW)
     
     fld = coyote_field2(state.panel,$
                         LABELFONT=state.buttonfont,$
                         FIELDFONT=state.textfont,$
                         TITLE='S/N Cut (0 means no cut):',$
                         UVALUE='S/N Cut',$
                         VALUE=state.snrcut,$
                         XSIZE=6,$
                         EVENT_PRO='xvspec_event',$
                         /CR_ONLY,$
                         TEXTID=textid)
     state.tmp_fld = [fld,textid]
     
     button = widget_button(state.panel,$
                            FONT=state.buttonfont,$
                            EVENT_PRO='xvspec_event',$
                            VALUE=' Done ',$
                            UVALUE='Done S/N Cut Button')

     mc_setfocus,state.tmp_fld


  endif
  
  if keyword_set(SMOOTH) then begin

     widget_control, state.but_smooth,/SET_BUTTON
     
     state.panel = widget_base(state.xvspec_base,$
                               /BASE_ALIGN_CENTER,$
                               FRAME=2,$
                               /ROW)
     
     fld = coyote_field2(state.panel,$
                         LABELFONT=state.buttonfont,$
                         FIELDFONT=state.textfont,$
                         TITLE='Gaussian FWHM:',$
                         UVALUE='Gaussian FWHM',$
                         VALUE=state.fwhm,$
                         XSIZE=10,$
                         EVENT_PRO='xvspec_event',$
                         /CR_ONLY,$
                         TEXTID=textid)
     state.tmp_fld = [fld,textid]
     
     button = widget_button(state.panel,$
                            FONT=state.buttonfont,$
                            EVENT_PRO='xvspec_event',$
                            VALUE=' Done ',$
                            UVALUE='Done Smoothing Button')

     mc_setfocus,state.tmp_fld
     
  endif

  if keyword_set(ASCII) or keyword_set(FITS) then begin

     uvalue = (keyword_set(ASCII) eq 1) ? $
              'Write ASCII Filename':'Write FITS Filename'

     state.panel = widget_base(state.xvspec_base,$
                               /BASE_ALIGN_CENTER,$
                               FRAME=2,$
                               /ROW)
     
     fld = coyote_field2(state.panel,$
                         LABELFONT=state.buttonfont,$
                         FIELDFONT=state.textfont,$
                         TITLE='Filename (sans suffix):',$
                         UVALUE=uvalue,$
                         XSIZE=15,$
                         EVENT_PRO='xvspec_event',$
                         /CR_ONLY,$
                         TEXTID=textid)
     state.tmp_fld = [fld,textid]
     
     button = widget_button(state.panel,$
                            FONT=state.buttonfont,$
                            EVENT_PRO='xvspec_event',$
                            VALUE=' Close ',$
                            UVALUE='Close Write Button')
     
     mc_setfocus,state.tmp_fld
        
  endif
  
end
;
;===============================================================================
;
pro xvspec_mkwidget,POSITION=position,GROUP_LEADER=group_leader,CANCEL=cancel

  common xvspec_state

  cleanplot,/SILENT

  
;  Build the widget.
  
  state.xvspec_base = widget_base(TITLE='xvspec', $
                                  /COLUMN,$
                                  MBAR=mbar,$
                                  /TLB_SIZE_EVENTS,$
                                  GROUP_LEADER=group_leader)

  button = widget_button(mbar, $
                         VALUE='File', $
                         /MENU,$
                         FONT=state.buttonfont)

     sbutton = widget_button(button, $
                             VALUE='Load FITS',$
                             UVALUE='Load Spextool FITS Button',$
                             EVENT_PRO='xvspec_event',$
                             FONT=state.buttonfont)

     sbutton = widget_button(button, $
                             VALUE='Load User Lines',$
                             UVALUE='Load User Lines Button',$
                             EVENT_PRO='xvspec_event',$
                             FONT=state.buttonfont)
     
     sbutton = widget_button(button, $
                             VALUE='View FITS Header',$
                             UVALUE='View FITS Header Button',$
                             EVENT_PRO='xvspec_event',$
                             FONT=state.buttonfont)

     state.but_writefits = widget_button(button, $
                                         VALUE='Write Spextool FITS File',$
                                         UVALUE='Write Spextool FITS File',$
                                         EVENT_PRO='xvspec_event',$
                                         FONT=state.buttonfont)
     
     state.but_writeascii = widget_button(button, $
                                          VALUE='Write ASCII File',$
                                          UVALUE='Write ASCII File',$
                                          EVENT_PRO='xvspec_event',$
                                          FONT=state.buttonfont)
     
     sbutton = widget_button(button, $
                             VALUE='Quit',$
                             UVALUE='Quit',$
                             EVENT_PRO='xvspec_event',$
                             FONT=state.buttonfont)

  button = widget_button(mbar, $
                         VALUE='Mode', $
                         /MENU,$
                         FONT=state.buttonfont)

     state.but_ladder = widget_button(button, $
                                      VALUE='Ladder Plot',$
                                      UVALUE='Ladder Plot Button',$
                                      EVENT_PRO='xvspec_event',$
                                      /CHECKED_MENU,$
                                      FONT=state.buttonfont)
     
     state.but_continuous = widget_button(button, $
                                          VALUE='Continuous Plot',$
                                          UVALUE='Continuous Plot Button',$
                                          EVENT_PRO='xvspec_event',$
                                          /CHECKED_MENU,$
                                          FONT=state.buttonfont)

     state.but_buffer = widget_button(mbar, $
                                      VALUE='Buffer', $
                                      /MENU,$
                                      FONT=state.buttonfont)
     
     state.but_buffera = widget_button(state.but_buffer, $
                                         VALUE='Buffer A',$
                                         UVALUE='Buffer A',$
                                         EVENT_PRO='xvspec_event',$
                                         /CHECKED_MENU,$
                                       FONT=state.buttonfont)

     state.but_bufferb = widget_button(state.but_buffer, $
                                         VALUE='Buffer B',$
                                         UVALUE='Buffer B',$
                                         EVENT_PRO='xvspec_event',$
                                         /CHECKED_MENU,$
                                         FONT=state.buttonfont)
     widget_control, state.but_buffera, /SET_BUTTON
     
  button = widget_button(mbar, $
                         VALUE='Spectrum', $
                         EVENT_PRO='xvspec_event',$
                         UVALUE='Spectrum Button',$
                         /MENU,$
                         /NO_RELEASE,$
                         FONT=state.buttonfont)
                
     state.but_flx = widget_button(button,$
                                     VALUE='Flux',$
                                     UVALUE='Plot Flux Button',$
                                     FONT=state.buttonfont,$
                                     /CHECKED_MENU)  
     widget_control, state.but_flx,/SET_BUTTON

     state.but_unc = widget_button(button,$
                                     VALUE='Uncertainty',$
                                     UVALUE='Plot Uncertainty Button',$
                                     FONT=state.buttonfont,$
                                     /CHECKED_MENU)  

     state.but_snrcut = widget_button(button,$
                                      VALUE='S/N',$
                                      UVALUE='Plot S/N Button',$
                                      FONT=state.buttonfont,$
                                      /CHECKED_MENU)  

  state.but_order = widget_button(mbar, $
                                  VALUE='Order', $
                                  UVALUE='Order Menu',$
                                  EVENT_PRO='xvspec_event',$
                                  /MENU,$
                                  FONT=state.buttonfont)

     *state.but_ord = widget_button(state.but_order,$
                                    VALUE='1',$
                                    UVALUE='Order Menu',$
                                    FONT=state.buttonfont)
  
  state.but_aperture = widget_button(mbar, $
                                     VALUE='Aperture', $
                                     UVALUE='Aperture Menu',$
                                     EVENT_PRO='xvspec_event',$
                                     /MENU,$
                                     FONT=state.buttonfont)

     *state.but_ap = widget_button(state.but_aperture,$
                                    VALUE='1',$
                                    UVALUE='Aperture Menu',$
                                    FONT=state.buttonfont,$
                                    /CHECKED_MENU)  
  
  button = widget_button(mbar, $
                         VALUE='Units', $
                         UVALUE='Units Button',$
                         /MENU,$
                         FONT=state.buttonfont)

     state.but_wavelength = widget_button(button, $
                                          VALUE='Wavelength', $
                                          EVENT_PRO='xvspec_event',$
                                          UVALUE='Wavelength Units Menu',$
                                          /MENU,$
                                          FONT=state.buttonfont)
     
        state.but_waveu[0] = widget_button(state.but_wavelength, $
                                           VALUE='um', $
                                           UVALUE='Wavelength Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)  
        
        state.but_waveu[1] = widget_button(state.but_wavelength, $
                                           VALUE='nm', $
                                           UVALUE='Wavelength Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)  
        
        state.but_waveu[2] = widget_button(state.but_wavelength, $
                                           VALUE='A', $
                                           UVALUE='Wavelength Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)  
        
     state.but_flux = widget_button(button, $
                                    VALUE='Flux', $
                                    EVENT_PRO='xvspec_event',$
                                    UVALUE='Flux Units Menu',$
                                    /MENU,$
                                    FONT=state.buttonfont)
     
        state.but_fluxu[0] = widget_button(state.but_flux, $
                                           VALUE='W m-2 um-1', $
                                           UVALUE='Flux Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)
        
        state.but_fluxu[1] = widget_button(state.but_flux, $
                                           VALUE='ergs s-1 cm-2 A-1', $
                                           UVALUE='Flux Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)
        
        state.but_fluxu[2] = widget_button(state.but_flux, $
                                           VALUE='W m-2 Hz-1', $
                                           UVALUE='Flux Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)     
        
        state.but_fluxu[3] = widget_button(state.but_flux, $
                                           VALUE='ergs s-1 cm-2 Hz-1', $
                                           UVALUE='Flux Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)  
        
        state.but_fluxu[4] = widget_button(state.but_flux, $
                                           VALUE='Jy', $
                                           UVALUE='Flux Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)
        
        state.but_fluxu[5] = widget_button(state.but_flux, $
                                           VALUE='W m-2', $
                                           UVALUE='Flux Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)
        
        state.but_fluxu[6] = widget_button(state.but_flux, $
                                           VALUE='ergs s-1 cm-2', $
                                           UVALUE='Flux Units Menu',$
                                           FONT=state.buttonfont,$
                                           /CHECKED_MENU)  
        
  button = widget_button(mbar, $
                         EVENT_PRO='xvspec_event',$
                         VALUE='Flags', $
                         UVALUE='Flags Button',$
                         /MENU,$
                         FONT=state.buttonfont)

     state.but_flags[0] = widget_button(button, $
                                        VALUE='Lincor Max Pixel (red)', $
                                        UVALUE='Saturated Pixel Menu',$
                                        FONT=state.buttonfont,$
                                        /CHECKED_MENU)
  
     state.but_flags[1] = widget_button(button, $
                                        VALUE='Replaced Pixel (blue)', $
                                        UVALUE='Replaced Pixel Menu',$
                                        FONT=state.buttonfont,$
                                          /CHECKED_MENU)

     state.but_flags[2] = widget_button(button, $
                                        VALUE='Fixed Pixel (cyan)', $
                                        UVALUE='Fixed Pixel Menu',$
                                        FONT=state.buttonfont,$
                                        /CHECKED_MENU)

     state.but_flags[3] = widget_button(button, $
                                        VALUE='Opt Extract Fail (yellow)', $
                                        UVALUE='Opt Extract Fail Menu',$
                                        FONT=state.buttonfont,$
                                        /CHECKED_MENU)
        
  button = widget_button(mbar, $
                         VALUE='Plot', $
                         EVENT_PRO='xvspec_event',$
                         UVALUE='Plot Button',$
                         /MENU,$
                         FONT=state.buttonfont)

     state.but_xlog = widget_button(button,$
                                    VALUE='X Log',$
                                    UVALUE='X Log Button',$
                                    FONT=state.buttonfont,$
                                    /CHECKED_MENU)  
     
     state.but_ylog = widget_button(button,$
                                    VALUE='Y Log',$
                                    UVALUE='Y Log Button',$
                                    FONT=state.buttonfont,$
                                    /CHECKED_MENU)

     state.but_xranges = widget_button(button,$
                                       VALUE='X Ranges',$
                                       UVALUE='X Ranges Button',$
                                       FONT=state.buttonfont,$
                                       /CHECKED_MENU)  
     
     state.but_3coloraltbut = widget_button(button,$
                                            VALUE='3-Color Alternate Spectra',$
                                    UVALUE='3-Color Alternate Spectra Button',$
                                            FONT=state.buttonfont,$
                                            /CHECKED_MENU)

     state.but_2coloraltbut = widget_button(button,$
                                            VALUE='2-Color Alternate Spectra',$
                                    UVALUE='2-Color Alternate Spectra Button',$
                                            FONT=state.buttonfont,$
                                            /CHECKED_MENU)
     
     state.but_plotatmos = widget_button(button,$
                                         VALUE='Plot Atmosphere',$
                                         UVALUE='Plot Atmosphere Button',$
                                         FONT=state.buttonfont,$
                                         /CHECKED_MENU)

     state.but_plothlines = widget_button(button,$
                                          VALUE='Plot Hydrogen Lines',$
                                          UVALUE='Plot Hydrogen Lines Button',$
                                          FONT=state.buttonfont,$
                                          /CHECKED_MENU)

     state.but_plotuserlines = widget_button(button,$
                                             VALUE='Plot User Lines',$
                                             UVALUE='Plot User Lines Button',$
                                             FONT=state.buttonfont,$
                                             /CHECKED_MENU)            

  button = widget_button(mbar, $
                         VALUE='Tools', $
                         UVALUE='Tools Button',$
                         EVENT_PRO='xvspec_event',$
                         /MENU,$
                         FONT=state.buttonfont)

     state.but_smooth = widget_button(button,$
                                      VALUE='Smooth',$
                                      UVALUE='Smooth Button',$
                                      FONT=state.buttonfont)

     state.but_snr = widget_button(button,$
                                   VALUE='S/N Cut',$
                                      UVALUE='S/N Cut Button',$
                                      FONT=state.buttonfont)
     
          
  button = widget_button(mbar, $
                         VALUE='Help', $
                         EVENT_PRO='xvspec_event',$
                         UVALUE='Help Button',$
                         /HELP,$
                         FONT=state.buttonfont)

  state.but_help = widget_button(button,$
                                 VALUE='xvspec Help',$
                                 UVALUE='Help Button',$
                                 FONT=state.buttonfont)
  
  row = widget_base(state.xvspec_base,$
                    YPAD=0,$
                    XPAD=0,$
                    /BASE_ALIGN_CENTER,$
                    /ROW)

     button = widget_button(row, $
                            VALUE='Load FITS', $
                            EVENT_PRO='xvspec_event',$
                            UVALUE='Load Spextool FITS Button',$
                            FONT=state.buttonfont)

     blank = widget_label(row,$
                          VALUE='')

     subrow = widget_base(row,$
                          /ROW,$
                          /TOOLBAR,$
                          /EXCLUSIVE)
         
        state.mbut_ladder = widget_button(subrow, $
                                          VALUE='Ladder', $
                                          EVENT_PRO='xvspec_event',$
                                          UVALUE='Ladder Plot Button',$
                                          /NO_RELEASE,$
                                          FONT=state.buttonfont)
            
        state.mbut_continuous = widget_button(subrow, $
                                              VALUE='Continuous', $
                                              EVENT_PRO='xvspec_event',$
                                              UVALUE='Continuous Plot Button',$
                                              /NO_RELEASE,$
                                              FONT=state.buttonfont)

     blank = widget_label(row,$
                          VALUE='')


     subrow = widget_base(row,$
                          /ROW,$
                          /TOOLBAR,$
                          /EXCLUSIVE)
     
        state.mbut_buffera = widget_button(subrow, $
                                           VALUE=' A ', $
                                           EVENT_PRO='xvspec_event',$
                                           UVALUE='Buffer A',$
                                           /NO_RELEASE,$
                                           FONT=state.buttonfont)
        widget_control, state.mbut_buffera,/SET_BUTTON
     
        
        state.mbut_bufferb = widget_button(subrow, $
                                           VALUE=' B ', $
                                           EVENT_PRO='xvspec_event',$
                                           UVALUE='Buffer B',$
                                           /NO_RELEASE,$
                                           FONT=state.buttonfont)
        widget_control, state.mbut_bufferb,SENSITIVE=0
        
      blank = widget_label(row,$
                           VALUE='')
            
      subrow = widget_base(row,$
                           /ROW,$
                           /TOOLBAR,$
                           /EXCLUSIVE)
      
         state.mbut_flx = widget_button(subrow, $
                                        VALUE='Flux', $
                                        EVENT_PRO='xvspec_event',$
                                        UVALUE='Plot Flux Button',$
                                        /NO_RELEASE,$
                                        FONT=state.buttonfont)
         
         state.mbut_unc = widget_button(subrow, $
                                        VALUE='Uncertainty', $
                                        EVENT_PRO='xvspec_event',$
                                        UVALUE='Plot Uncertainty Button',$
                                        /NO_RELEASE,$
                                        FONT=state.buttonfont)
         
         state.mbut_snr = widget_button(subrow, $
                                        VALUE='S/N', $
                                        EVENT_PRO='xvspec_event',$
                                        UVALUE='Plot S/N Button',$
                                        /NO_RELEASE,$
                                        FONT=state.buttonfont)
         widget_control, state.mbut_flx,/SET_BUTTON

      blank = widget_label(row,$
                           VALUE='')
         
      subrow = widget_base(row,$
                           /ROW,$
                           /TOOLBAR,$
                           /NONEXCLUSIVE)
      
         state.mbut_plotatmos = widget_button(subrow, $
                                              VALUE='Atmosphere', $
                                              EVENT_PRO='xvspec_event',$
                                              UVALUE='Plot Atmosphere Button',$
                                              FONT=state.buttonfont)

      subrow = widget_base(row,$
                           /ROW,$
                           /TOOLBAR,$
                           /EXCLUSIVE)

         state.mbut_2color = widget_button(subrow, $
                                           VALUE='2 Color', $
                                           EVENT_PRO='xvspec_event',$
                                     UVALUE='2-Color Alternate Spectra Button',$
                                           FONT=state.buttonfont)               
      
         state.mbut_3color = widget_button(subrow, $
                                           VALUE='3 Color', $
                                           EVENT_PRO='xvspec_event',$
                                     UVALUE='3-Color Alternate Spectra Button',$
                                           FONT=state.buttonfont)         
         
      button = widget_button(row, $
                             VALUE='Quit',$
                             UVALUE='Quit',$
                             EVENT_PRO='xvspec_event',$
                             FONT=state.buttonfont)
      
      state.message = widget_text(state.xvspec_base, $
                                  VALUE='',$
                                  YSIZE=1)

      state.plotbase = widget_base(state.xvspec_base,$
                                   /ROW)
      
  if state.mode eq 'Continuous' then begin
    
     state.plotwin = widget_draw(state.plotbase,$
                                 /ALIGN_CENTER,$
                                 XSIZE=state.scroll_size[0],$
                                 YSIZE=state.scroll_size[1],$
                                 EVENT_PRO='xvspec_plotwinevent',$
                                 /MOTION_EVENTS,$
                                 /KEYBOARD_EVENTS,$
                                 /BUTTON_EVENTS,$
                                 /TRACKING_EVENTS)
     

     state.ranges = widget_base(state.xvspec_base,$
                                /ROW,$
                                /BASE_ALIGN_LEFT,$
                                FRAME=2)

        xmin = coyote_field2(state.ranges,$
                             LABELFONT=state.buttonfont,$
                             FIELDFONT=state.textfont,$
                             TITLE='X Min:',$
                             UVALUE='X Min',$
                             XSIZE=12,$
                             EVENT_PRO='xvspec_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.xmin_fld = [xmin,textid]
                
        xmax = coyote_field2(state.ranges,$
                             LABELFONT=state.buttonfont,$
                             FIELDFONT=state.textfont,$
                             TITLE='X Max:',$
                             UVALUE='X Max',$
                             XSIZE=12,$
                             EVENT_PRO='xvspec_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.xmax_fld = [xmax,textid]
        
        ymin = coyote_field2(state.ranges,$
                             LABELFONT=state.buttonfont,$
                             FIELDFONT=state.textfont,$
                             TITLE='Y Min:',$
                             UVALUE='Y Min',$
                             XSIZE=12,$
                             EVENT_PRO='xvspec_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.ymin_fld = [ymin,textid]
        
        ymax = coyote_field2(state.ranges,$
                             LABELFONT=state.buttonfont,$
                             FIELDFONT=state.textfont,$
                             TITLE='Y Max:',$
                             UVALUE='Y Max',$
                             XSIZE=12,$
                             EVENT_PRO='xvspec_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.ymax_fld = [ymax,textid]

     state.slider = widget_slider(state.xvspec_base,$
                                  UVALUE='Slider',$
                                  EVENT_PRO='xvspec_event',$
                                  /DRAG,$
                                  /SUPPRESS_VALUE,$
                                  FONT=state.buttonfont)
     widget_control, state.slider, SET_VALUE=state.sliderval

        
  endif

  if state.mode eq 'Ladder' and state.scrollbars then begin
     

     state.plotwin = widget_draw(state.plotbase,$
                                 /ALIGN_CENTER,$
                                 XSIZE=state.plot_size[0],$
                                 YSIZE=state.plot_size[1],$
                                 X_SCROLL_SIZE=state.scroll_size[0],$
                                 Y_SCROLL_SIZE=state.scroll_size[1],$
                                 /SCROLL,$
                                 EVENT_PRO='xvspec_plotwinevent',$
                                 /KEYBOARD_EVENTS,$
                                 /MOTION_EVENTS,$
                                 /BUTTON_EVENTS,$
                                 /TRACKING_EVENTS)

  endif

  if state.mode eq 'Ladder' and ~state.scrollbars then begin
    
     state.plotwin = widget_draw(state.plotbase,$
                                 /ALIGN_CENTER,$
                                 XSIZE=state.plot_size[0],$
                                 YSIZE=state.plot_size[1],$
                                 EVENT_PRO='xvspec_plotwinevent',$
                                 /KEYBOARD_EVENTS,$
                                 /BUTTON_EVENTS,$
                                 /MOTION_EVENTS,$
                                 /TRACKING_EVENTS)
     

  endif

     
; Get things running.  Center the widget using the Fanning routine.

   if n_elements(POSITION) eq 0 then position = [0.5,0.5]
   cgcentertlb,state.xvspec_base,position[0],position[1]

   widget_control, state.xvspec_base, /REALIZE
   
;  Get plotwin ids

   widget_control, state.plotwin, GET_VALUE=x
   state.plotwin_wid=x
   window, /FREE, /PIXMAP,XSIZE=state.plot_size[0],YSIZE=state.plot_size[1]
   state.pixmap_wid = !d.window
   
;  Get sizes for things.
   
   widget_geom = widget_info(state.xvspec_base, /GEOMETRY)

   state.winbuffer[0]=widget_geom.xsize-state.scroll_size[0]
   state.winbuffer[1]=widget_geom.ysize-state.scroll_size[1]
   
; Start the Event Loop. This will be a non-blocking program.
   
   XManager, 'xvspec', $
             state.xvspec_base, $
             /NO_BLOCK,$
             EVENT_HANDLER='xvspec_resizeevent',$
             CLEANUP='xvspec_cleanup'
  
end
;
;===============================================================================
;
pro xvspec_cleanup,base
  
  common xvspec_state

  if n_elements(state) ne 0 then begin

     ptr_free, state.ahdr
     ptr_free, state.amspectra
     ptr_free, state.aspectra
     ptr_free, state.atrans
     ptr_free, state.awave
     ptr_free, state.bhdr
     ptr_free, state.bspectra
     ptr_free, state.bmspectra
     ptr_free, state.but_ap
     ptr_free, state.but_ord
     ptr_free, state.flxyranges
     ptr_free, state.funits
     ptr_free, state.orders
     ptr_free, state.pspectra
     ptr_free, state.pyranges
     ptr_free, state.snryranges
     ptr_free, state.uncyranges
     ptr_free, state.wranges
     ptr_free, state.xranges
     ptr_free, state.wunits
     ptr_free, state.userlines
     ptr_free, state.usernames
     
  endif
  state = 0B
  
end
;
;===============================================================================
;
pro xvspec_convertflux

  common xvspec_state

;  First convert each order
  
  for i = 0,state.norders-1 do begin
     
     for j = 0,state.naps-1 do begin

;  Convert the 'a' spectra wavelengths
        
        x = (*state.aspectra)[*,0,i*state.naps+j]
        y = (*state.aspectra)[*,1,i*state.naps+j]
        e = (*state.aspectra)[*,2,i*state.naps+j]
        
        ny = mc_chfunits(x,y,(*state.wunits)[0],(*state.funits)[0],$
                         (*state.funits)[1],IERROR=e,OERROR=oe,CANCEL=cancel)
        if cancel then return
        
        (*state.aspectra)[*,1,i*state.naps+j] = ny
        (*state.aspectra)[*,2,i*state.naps+j] = oe

        x = (*state.amspectra)[*,0,i*state.naps+j]
        y = (*state.amspectra)[*,1,i*state.naps+j]
        e = (*state.amspectra)[*,2,i*state.naps+j]
        
        ny = mc_chfunits(x,y,(*state.wunits)[0],(*state.funits)[0],$
                         (*state.funits)[1],IERROR=e,OERROR=oe,CANCEL=cancel)
        if cancel then return

        (*state.amspectra)[*,1,i*state.naps+j] = ny
        (*state.amspectra)[*,2,i*state.naps+j] = oe

;  Conver the pyranges

        ny = mc_chfunits((*state.wranges)[*,i],(*state.pyranges)[*,i], $
                         (*state.wunits)[0],(*state.funits)[0],$
                         (*state.funits)[1],CANCEL=cancel)
        if cancel then return
        (*state.pyranges)[*,i] = ny
                
;  Convert the 'b' spectra wavelengths if necessary
        
        if state.nbuffers eq 2 then begin

           x = (*state.bspectra)[*,0,i*state.naps+j]
           y = (*state.bspectra)[*,1,i*state.naps+j]
           e = (*state.bspectra)[*,2,i*state.naps+j]
           
           ny = mc_chfunits(x,y,(*state.wunits)[0],(*state.funits)[0],$
                            (*state.funits)[1],IERROR=e,OERROR=oe, $
                            CANCEL=cancel)
           if cancel then return
           (*state.bspectra)[*,1,i*state.naps+j] = ny
           (*state.bspectra)[*,2,i*state.naps+j] = oe
           
           x = (*state.bmspectra)[*,0,i*state.naps+j]
           y = (*state.bmspectra)[*,1,i*state.naps+j]
           e = (*state.bmspectra)[*,2,i*state.naps+j]
           
           ny = mc_chfunits(x,y,(*state.wunits)[0],(*state.funits)[0],$
                            (*state.funits)[1],IERROR=e,OERROR=oe, $
                            CANCEL=cancel)
           if cancel then return
           (*state.bmspectra)[*,1,i*state.naps+j] = ny
           (*state.bmspectra)[*,2,i*state.naps+j] = oe

        endif
        
     endfor
     
  endfor
  
  case state.buffer of
     
     0: *state.pspectra = *state.amspectra
     1: *state.pspectra = *state.bmspectra
     
  endcase
  
;  Convert the flxyranges and uncyranges (snryranges remains constant)

  ny = mc_chfunits(state.flxwrange,state.flxyrange,(*state.wunits)[0], $
                   (*state.funits)[0],(*state.funits)[1],CANCEL=cancel)
  if cancel then return
  state.flxyrange = ny

  ny = mc_chfunits(state.flxwrange,state.uncyrange,(*state.wunits)[0], $
                   (*state.funits)[0],(*state.funits)[1],CANCEL=cancel)
  if cancel then return
  state.uncyrange = ny

;  Convert the absflxyranges and absuncyranges (snryranges remains constant)

  ny = mc_chfunits(state.absflxwrange,state.flxyrange,(*state.wunits)[0], $
                   (*state.funits)[0],(*state.funits)[1],CANCEL=cancel)
  if cancel then return
  state.absflxyrange = ny

  ny = mc_chfunits(state.absflxwrange,state.uncyrange,(*state.wunits)[0], $
                   (*state.funits)[0],(*state.funits)[1],CANCEL=cancel)
  if cancel then return
  state.absuncyrange = ny
  
;  Convert the pyrange and pabsyrange 

  ny = mc_chfunits(state.pwrange,state.pyrange,(*state.wunits)[0], $
                   (*state.funits)[0],(*state.funits)[1],CANCEL=cancel)
  if cancel then return
  state.pyrange = ny

  ny = mc_chfunits(state.pwrange,state.pabsyrange,(*state.wunits)[0], $
                   (*state.funits)[0],(*state.funits)[1],CANCEL=cancel)
  if cancel then return
  state.pabsyrange = ny

;  Update units and plot titles
  
  ftitle = mc_getfunits((*state.funits)[1],CANCEL=cancel)
  if cancel then return

  lidx = strpos(ftitle,'(')
  ridx = strpos(ftitle,')')
  ypunits = strmid(ftitle,lidx+1,ridx-lidx-1)
  state.ytitle  = [ftitle,'!5Uncertainty ('+ypunits+')','!5S/N']

  case state.spectype of

     'Flux': state.pytitle = state.ytitle[0]
     'Uncertainty': state.pytitle = state.ytitle[0]
     'S/N': state.pytitle = state.ytitle[0]

  endcase
  
  *state.funits = (*state.funits)[1]

end
;
;===============================================================================
;
pro xvspec_convertwave

  common xvspec_state

  mc_getwunits,(*state.wunits)[1],wunits,xtitle,CANCEL=cancel
  if cancel then return
  
  state.xtitle = xtitle

;  First convert each order
  
  for i = 0,state.norders-1 do begin
     
     for j = 0,state.naps-1 do begin

;  Convert the 'a' spectra wavelengths
        
        x = (*state.aspectra)[*,0,i*state.naps+j]
        nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                         CANCEL=cancel)
        if cancel then return
        (*state.aspectra)[*,0,i*state.naps+j] = nx
        (*state.amspectra)[*,0,i*state.naps+j] = nx

;  Convert the wranges array
        
        x = (*state.wranges)[*,i*state.naps+j]
        nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                         CANCEL=cancel)
        if cancel then return
        (*state.wranges)[*,i*state.naps+j] = nx

;  Convert the 'b' spectra wavelengths if need be
        
        if state.nbuffers eq 2 then begin

        x = (*state.bspectra)[*,0,i*state.naps+j]
        nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                         CANCEL=cancel)
        if cancel then return
        (*state.bspectra)[*,0,i*state.naps+j] = nx
        (*state.bmspectra)[*,0,i*state.naps+j] = nx
           
        endif
        
     endfor
     
  endfor

  case state.buffer of

     0: *state.pspectra = *state.amspectra
     1: *state.pspectra = *state.bmspectra
     
  endcase

;  Convert the plot ranges

  x = state.flxwrange
  nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                   CANCEL=cancel)
  if cancel then return
  state.flxwrange = nx

  x = state.absflxwrange
  nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                   CANCEL=cancel)
  if cancel then return
  state.absflxwrange = nx

  x = state.uncwrange
  nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                   CANCEL=cancel)
  if cancel then return
  state.uncwrange = nx

  x = state.absuncwrange
  nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                   CANCEL=cancel)
  if cancel then return
  state.absuncwrange = nx

    x = state.snrwrange
  nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                   CANCEL=cancel)
  if cancel then return
  state.snrwrange = nx

  x = state.abssnrwrange
  nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                   CANCEL=cancel)
  if cancel then return
  state.abssnrwrange = nx
  
  x = state.pwrange
  nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                   CANCEL=cancel)
  if cancel then return
  state.pwrange = nx

  x = state.pabswrange
  nx = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1], $
                   CANCEL=cancel)
  if cancel then return
  state.pabswrange = nx

;  Convert the atmosphere if necessary

  If n_elements(*state.awave) ne 0 then begin
  
     x = *state.awave
     nx = mc_chwunits(*state.awave,(*state.wunits)[0],(*state.wunits)[1], $
                      CANCEL=cancel)
     if cancel then return
     
     *state.awave = nx

  endif

;  Convert the hydrogen lines

  x = state.hlines
  x = mc_chwunits(x,(*state.wunits)[0],(*state.wunits)[1],CANCEL=cancel)
  if cancel then return
  state.hlines = x

;  Change units
  
  *state.wunits = (*state.wunits)[1] 
     
end
;
;===============================================================================
;
pro xvspec_getranges

  common xvspec_state

;  Do it for the afile (bfile comes along for the ride)
  
  spectra = *state.amspectra

  for i = 0,state.norders-1 do begin

     cwave  = reform(spectra[*,0,i*state.naps+state.ap])
     cflx  = reform(spectra[*,1,i*state.naps+state.ap])
     cunc = reform(spectra[*,2,i*state.naps+state.ap])
     csnr = cflx/cunc

;  Do each in turn

;  Do the wavelengths ignoring NaNs
     
     z = where(finite(cflx) eq 1)
     (*state.wranges)[*,i] = [min(cwave[z],MAX=max,/NAN),max]

;  Do the flux, first smooth to avoid bad pixels
     
     x = findgen(n_elements(cflx))
     smooth = mc_robustsg(x,cflx,5,3,0.1,CANCEL=cancel)
     if cancel then return
     
     min = min(smooth[*,1],/NAN,MAX=max)
     (*state.flxyranges)[*,i] = mc_bufrange([min,max],0.05)

;  Do the unc, first smooth to avoid bad pixels

     smooth = mc_robustsg(x,cunc,5,3,0.1,CANCEL=cancel)
     if cancel then return
     
     min = min(smooth[*,1],/NAN,MAX=max)
     (*state.uncyranges)[*,i] = mc_bufrange([min,max],0.05)

;  Do the unc, first smooth to avoid bad pixels
     
     smooth = mc_robustsg(x,csnr,5,3,0.1,CANCEL=cancel)
     if cancel then return
     
     min = min(smooth[*,1],/NAN,MAX=max)
     (*state.snryranges)[*,i] = mc_bufrange([min,max],0.05)
     
  endfor

;  Do it for the continuous case
  
  state.flxwrange = [min(*state.wranges,MAX=max),max]
  state.flxyrange = [min(*state.flxyranges,MAX=max),max]
  state.absflxwrange = state.flxwrange
  state.absflxyrange = state.flxyrange
  
  state.uncwrange = state.flxwrange
  state.uncyrange = [min(*state.uncyranges,MAX=max),max]
  state.absuncwrange = state.uncwrange
  state.absuncyrange = state.uncyrange

  state.snrwrange = state.flxwrange
  state.snryrange = [min(*state.snryranges,MAX=max),max]
  state.abssnrwrange = state.snrwrange
  state.abssnryrange = state.snryrange

end
;
;===============================================================================
;
pro xvspec_loadspectra,afile,bfile,WIDGET_ID=widget_id,CANCEL=cancel

  cancel = 0

  common xvspec_state

  if n_elements(afile) eq 0 then begin
     
     x        = findgen(1000)/10.
     y        = exp(-0.04*x)*sin(x)
     y        = y-min(y)+100
     e        = sqrt(y)
     f        = fltarr(1000)+0.0
     aspectra  = [[x],[y],[e],[f]]
     ahdr      = ''
     aobsmode  = '1D'
     aSTART    = min(x)
     aSTOP     = max(x)
     anorders  = 1
     aorders   = 1
     anaps     = 1
     axtitle   = '!7k!5 (pixels)'
     aytitle   = '!5f (DN s!U-1!N)'
     afile    = 'tmp'
     arp       = 0
     axunits   = 'pixels'
     ayunits   = 'flux'
     *state.wunits = !values.f_nan
     *state.funits = !values.f_nan
     axranges   = [astart,astop]
     
  endif else begin
     
     mc_readspec,afile,aspectra,ahdr,aobsmode,astart,astop,anorders,anaps, $
                 aorders,axunits,ayunits,aslith_pix,aslith_arc,aslitw_pix, $
                 aslitw_arc,arp,aairmass,axtitle,aytitle,instr,axranges, $
                 /SILENT, WIDGET_ID=widget_id,CANCEL=cancel
     if cancel then return
     
;  Check for flag array

     aspectra = mc_caaspecflags(aspectra,CANCEL=cancel)
     if cancel then return
          
  endelse

;  Load the a buffer
  
  state.afile      = file_basename(afile)
  *state.ahdr      = ahdr
  *state.aspectra  = aspectra
  *state.amspectra = aspectra
  state.norders    = anorders
  state.naps       = anaps
  *state.orders    = (n_elements(AORDERS) ne 0) ? aorders:1
  state.ap         = 0
  state.nbuffers   = 1
  state.xunits     = axunits
  state.yunits     = ayunits

  *state.xranges    = axranges
  *state.wranges    = fltarr(2,anorders)
  *state.flxyranges = fltarr(2,anorders)
  *state.uncyranges = fltarr(2,anorders)
  *state.snryranges = fltarr(2,anorders)

;  Deal with the units
  
  state.xtitle = axtitle
  lidx = strpos(aytitle,'(')
  ridx = strpos(aytitle,')')
  ypunits = strmid(aytitle,lidx+1,ridx-lidx-1)
  state.ytitle  = [aytitle,'!5Uncertainty ('+ypunits+')','!5S/N']
   
  if n_elements(bfile) ne 0 then begin
     
     mc_readspec,bfile,bspectra,bhdr,bobsmode,bstart,bstop,bnorders,bnaps, $
                 borders,bxunits,byunits,bslith_pix,bslith_arc,bslitw_pix, $
                 bslitw_arc,arp,bairmass,bxtitle,bytitle,instr,bxranges, $
                 /SILENT,WIDGET_ID=widget_id,CANCEL=cancel
     if cancel then return
     
;  Check for flag array

     bspectra = mc_caaspecflags(bspectra,CANCEL=cancel)
     if cancel then return

;  If it matches perfectly, load the B frame
     
     if aobsmode eq bobsmode and anorders eq bnorders and anaps eq bnaps and $
        total(aorders) eq total(borders) and axunits eq bxunits and $
        ayunits eq byunits and aslith_pix eq bslith_pix and $
        aslith_arc eq bslith_arc and aslitw_pix eq bslitw_pix and $
        aslitw_arc eq bslitw_arc then begin

        state.bfile      = file_basename(bfile)
        *state.bhdr      = bhdr
        *state.bspectra  = bspectra
        *state.bmspectra = bspectra
        state.nbuffers   = 2

     endif 
     
  endif

;  Smooth data if necessary

  xvspec_smoothspectra

;  Load the atmospheric transmission if need be

  if arp ne 0 and (axunits eq 'um' or axunits eq 'nm' or axunits eq 'A') $
  then begin

;  Get the resolutions available
     
     files = file_basename(file_search(filepath('atran*.fits', $
                                                ROOT_DIR=state.spextoolpath, $
                                                SUBDIR='data')))
     
     nfiles = n_elements(files)
     rps = lonarr(nfiles)
     for i =0,nfiles-1 do rps[i] = long(strmid( $
        file_basename(files[i],'.fits'),5))
     
     min = min(abs(rps-arp),idx)
     
     spec = readfits(filepath('atran'+strtrim(rps[idx],2)+'.fits', $
                              ROOT_DIR=state.spextoolpath, $
                              SUBDIR='data'),/SILENT)

     *state.awave = reform(spec[*,0])
     *state.atrans = reform(spec[*,1])

  endif else begin

     *state.awave = 0
     *state.atrans = 0
     
  endelse
     
end
;
;===============================================================================
;
pro xvspec_modwin,CONTINUOUS=continuous,LADDER=ladder,PLOTWINSIZE=plotwinsize, $
                  MODE=mode,NOUPDATE=noupdate

  common xvspec_state

;  Get current status

  current = state.mode
  if current eq 'Ladder' then current = current+strtrim(state.scrollbars,2)

;  Get requested status

  request = state.mode

  if n_elements(PLOTWINSIZE) eq 0 or keyword_set(NOUPDATE) then begin
     
     plotwinsize = (current ne 'Ladder1') ? state.plot_size:state.scroll_size
     
  endif else begin

;  Get screensize
  
     screensize = get_screen_size()
     plotwinsize = screensize*plotwinsize
     
  endelse

  if keyword_set(MODE) then begin
     
     case mode of
        
        'Continuous': request = 'Continuous'
        
        'Ladder': request = 'Ladder'
        
        else:  begin
           
           ok = dialog_message([['Unidentified view mode requested.'],$
                                ['Defaulting to Continuous.']],/INFO, $
                                DIALOG_PARENT=state.xvspec_base)

           request = 'Continuous'
           
        end
        
     endcase

  endif

  if keyword_set(CONTINUOUS) then request = 'Continuous'
  if keyword_set(LADDER) then request = 'Ladder'
  if keyword_set(NOUPDATE) then request = state.mode

  state.scroll_size[0] = plotwinsize[0]
  state.scroll_size[1] = plotwinsize[1]
  state.plot_size[0] = plotwinsize[0]
  state.plot_size[1] = plotwinsize[1]

  if request eq 'Ladder' then begin
  
     state.plot_size[1] = state.scroll_size[1]>state.pixperorder*state.norders
     state.scrollbars = state.plot_size[1] gt state.scroll_size[1]
     request = request+strtrim(state.scrollbars,2)

  endif

;  Now do the actual changes
  
  if current eq 'Continuous' then begin

     case request of

        'Continuous': begin

           widget_control, state.plotwin, DRAW_XSIZE=state.plot_size[0], $
                           DRAW_YSIZE=state.plot_size[1]           
           state.mode = 'Continuous'
           
        end

        'Ladder0': begin  ;  no scroll bars

           widget_control, state.xvspec_base,UPDATE=0
           widget_control, state.plotbase,UPDATE=0
           widget_control, state.ranges,/DESTROY
           widget_control, state.slider, /DESTROY
           widget_control, state.plotwin,/DESTROY
           if state.smoothingpanel then widget_control, state.panel,/DESTROY
           if state.filenamepanel then widget_control, state.panel,/DESTROY
           
           state.plotwin = widget_draw(state.plotbase,$
                                       /ALIGN_CENTER,$
                                       XSIZE=state.plot_size[0],$
                                       YSIZE=state.plot_size[1],$
                                       EVENT_PRO='xvspec_plotwinevent',$
                                       /KEYBOARD_EVENTS,$
                                       /BUTTON_EVENTS,$
                                       /MOTION_EVENTS,$
                                       /TRACKING_EVENTS)
                      

           if state.smoothingpanel then xvspec_addpanel,/SMOOTH
           if state.writefilepanel then xvspec_addpanel,/WRITE

           widget_control, state.plotbase,UPDATE=1
           widget_control, state.xvspec_base,UPDATE=1
           state.mode = 'Ladder'
           
        end

        'Ladder1': begin  ; scroll bars

           widget_control, state.xvspec_base,UPDATE=0
           widget_control, state.plotbase,UPDATE=0
           widget_control, state.ranges,/DESTROY
           widget_control, state.slider, /DESTROY
           widget_control, state.plotwin,/DESTROY
           if state.smoothingpanel then widget_control, state.panel,/DESTROY
           if state.filenamepanel then widget_control, state.panel,/DESTROY
           
           state.plotwin = widget_draw(state.plotbase,$
                                       /ALIGN_CENTER,$
                                       XSIZE=state.plot_size[0],$
                                       YSIZE=state.plot_size[1],$
                                       X_SCROLL_SIZE=state.scroll_size[0],$
                                       Y_SCROLL_SIZE=state.scroll_size[1],$
                                       /SCROLL,$
                                       EVENT_PRO='xvspec_plotwinevent',$
                                       /KEYBOARD_EVENTS,$
                                       /BUTTON_EVENTS,$
                                       /MOTION_EVENTS,$
                                       /TRACKING_EVENTS)

           if state.smoothingpanel then xvspec_addpanel,/SMOOTH
           if state.filenamepanel then xvspec_addpanel,/WRITE

           widget_control, state.plotbase,UPDATE=1           
           widget_control, state.xvspec_base,UPDATE=1
           state.mode = 'Ladder'
           
        end
     

     endcase
     
  endif

  if current eq 'Ladder0' then begin

     case request of

        'Continuous': begin

           widget_control, state.xvspec_base,UPDATE=0
           widget_control, state.plotbase,UPDATE=0
           
;           state.message = widget_text(state.xvspec_base, $
;                                       VALUE='',$
;                                       YSIZE=1)

           widget_control, state.plotwin, DRAW_XSIZE=state.plot_size[0], $
                           DRAW_YSIZE=state.plot_size[1],$
                           /DRAW_MOTION_EVENTS

           state.slider = widget_slider(state.plotbase,$
                                        UVALUE='Slider',$
                                        EVENT_PRO='xvspec_event',$
                                        /DRAG,$
                                        /SUPPRESS_VALUE,$
                                        FONT=state.buttonfont)
           state.sliderval = 50
           widget_control, state.slider, SET_VALUE=state.sliderval

           state.ranges = widget_base(state.xvspec_base,$
                                      /ROW,$
                                      /BASE_ALIGN_LEFT,$
                                      FRAME=2)
           
              xmin = coyote_field2(state.ranges,$
                                   LABELFONT=state.buttonfont,$
                                   FIELDFONT=state.textfont,$
                                   TITLE='X Min:',$
                                   UVALUE='X Min',$
                                   XSIZE=12,$
                                   EVENT_PRO='xvspec_minmaxevent',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.xmin_fld = [xmin,textid]
              
              xmax = coyote_field2(state.ranges,$
                                   LABELFONT=state.buttonfont,$
                                   FIELDFONT=state.textfont,$
                                   TITLE='X Max:',$
                                   UVALUE='X Max',$
                                   XSIZE=12,$
                                   EVENT_PRO='xvspec_minmaxevent',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.xmax_fld = [xmax,textid]
              
              ymin = coyote_field2(state.ranges,$
                                   LABELFONT=state.buttonfont,$
                                   FIELDFONT=state.textfont,$
                                   TITLE='Y Min:',$
                                   UVALUE='Y Min',$
                                   XSIZE=12,$
                                   EVENT_PRO='xvspec_minmaxevent',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.ymin_fld = [ymin,textid]
              
              ymax = coyote_field2(state.ranges,$
                                   LABELFONT=state.buttonfont,$
                                   FIELDFONT=state.textfont,$
                                   TITLE='Y Max:',$
                                   UVALUE='Y Max',$
                                   XSIZE=12,$
                                   EVENT_PRO='xvspec_minmaxevent',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.ymax_fld = [ymax,textid]

              widget_control, state.plotbase,UPDATE=1
              widget_control, state.xvspec_base,UPDATE=1
              
              state.mode = 'Continuous'
              
        end

        'Ladder0': begin

           widget_control, state.plotwin, DRAW_XSIZE=state.plot_size[0], $
                           DRAW_YSIZE=state.plot_size[1]           
           state.mode = 'Ladder'

        end

        'Ladder1': begin

           widget_control, state.xvspec_base,UPDATE=0
           widget_control, state.plotbase,UPDATE=0
           widget_control, state.plotwin,/DESTROY
           if state.smoothingpanel then widget_control, state.panel,/DESTROY
           if state.filenamepanel then widget_control, state.panel,/DESTROY
           
           state.plotwin = widget_draw(state.plotbase,$
                                       /ALIGN_CENTER,$
                                       XSIZE=state.plot_size[0],$
                                       YSIZE=state.plot_size[1],$
                                       X_SCROLL_SIZE=state.scroll_size[0],$
                                       Y_SCROLL_SIZE=state.scroll_size[1],$
                                       /SCROLL,$
                                       EVENT_PRO='xvspec_plotwinevent',$
                                       /KEYBOARD_EVENTS,$
                                       /MOTION_EVENTS,$
                                       /BUTTON_EVENTS,$
                                       /TRACKING_EVENTS)

           if state.smoothingpanel then xvspec_addpanel,/SMOOTH
           if state.filenamepanel then xvspec_addpanel,/WRITE

           widget_control, state.plotbase,UPDATE=1
           widget_control, state.xvspec_base,UPDATE=1
           state.mode = 'Ladder'
           
        end
        
     endcase

  endif

  if current eq 'Ladder1' then begin

     case request of

        'Continuous': begin

           widget_control, state.xvspec_base,UPDATE=0
           widget_control, state.plotbase,UPDATE=0           
           widget_control, state.plotwin,/DESTROY
           if state.smoothingpanel then widget_control, state.panel,/DESTROY
           if state.filenamepanel then widget_control, state.panel,/DESTROY

           state.plotwin = widget_draw(state.plotbase,$
                                       /ALIGN_BOTTOM,$
                                       XSIZE=state.plot_size[0],$
                                       YSIZE=state.plot_size[1],$
                                       EVENT_PRO='xvspec_plotwinevent',$
                                       /MOTION_EVENTS,$
                                       /KEYBOARD_EVENTS,$
                                       /BUTTON_EVENTS,$
                                       /TRACKING_EVENTS)

           state.ranges = widget_base(state.xvspec_base,$
                                      /ROW,$
                                      /BASE_ALIGN_LEFT,$
                                      FRAME=2)
           
              xmin = coyote_field2(state.ranges,$
                                   LABELFONT=state.buttonfont,$
                                   FIELDFONT=state.textfont,$
                                   TITLE='X Min:',$
                                   UVALUE='X Min',$
                                   XSIZE=12,$
                                   EVENT_PRO='xvspec_minmaxevent',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.xmin_fld = [xmin,textid]
              
              xmax = coyote_field2(state.ranges,$
                                   LABELFONT=state.buttonfont,$
                                   FIELDFONT=state.textfont,$
                                   TITLE='X Max:',$
                                   UVALUE='X Max',$
                                   XSIZE=12,$
                                   EVENT_PRO='xvspec_minmaxevent',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.xmax_fld = [xmax,textid]
              
              ymin = coyote_field2(state.ranges,$
                                   LABELFONT=state.buttonfont,$
                                   FIELDFONT=state.textfont,$
                                   TITLE='Y Min:',$
                                   UVALUE='Y Min',$
                                   XSIZE=12,$
                                   EVENT_PRO='xvspec_minmaxevent',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.ymin_fld = [ymin,textid]
           
              ymax = coyote_field2(state.ranges,$
                                   LABELFONT=state.buttonfont,$
                                   FIELDFONT=state.textfont,$
                                   TITLE='Y Max:',$
                                   UVALUE='Y Max',$
                                   XSIZE=12,$
                                   EVENT_PRO='xvspec_minmaxevent',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.ymax_fld = [ymax,textid]
                      
           state.slider = widget_slider(state.xvspec_base,$
                                        UVALUE='Slider',$
                                        EVENT_PRO='xvspec_event',$
                                        /DRAG,$
                                        /SUPPRESS_VALUE,$
                                        FONT=state.buttonfont)

           if state.smoothingpanel then xvspec_addpanel,/SMOOTH
           if state.filenamepanel then xvspec_addpanel,/WRITE

           state.sliderval = 50
           widget_control, state.slider, SET_VALUE=state.sliderval

           widget_control, state.plotbase,UPDATE=1
           widget_control, state.xvspec_base,UPDATE=1


          state.mode = 'Continuous'
           
        end

        'Ladder0': begin

           widget_control, state.xvspec_base,UPDATE=0
           widget_control, state.plotbase,UPDATE=0
           widget_control, state.plotwin,/DESTROY
           if state.smoothingpanel then widget_control, state.panel,/DESTROY
           if state.filenamepanel then widget_control, state.panel,/DESTROY
           
           state.plotwin = widget_draw(state.plotbase,$
                                       /ALIGN_CENTER,$
                                       XSIZE=state.plot_size[0],$
                                       YSIZE=state.plot_size[1],$
                                       EVENT_PRO='xvspec_plotwinevent',$
                                       /KEYBOARD_EVENTS,$
                                       /BUTTON_EVENTS,$
                                       /MOTION_EVENTS,$
                                       /TRACKING_EVENTS)

           if state.smoothingpanel then xvspec_addpanel,/SMOOTH
           if state.filenamepanel then xvspec_addpanel,/WRITE

           widget_control, state.plotbase,UPDATE=1           
           widget_control, state.xvspec_base,UPDATE=1
           state.mode = 'Ladder'
           
        end

        'Ladder1': begin

           widget_control, state.xvspec_base,UPDATE=0
           widget_control, state.plotbase,UPDATE=0
           widget_control, state.plotwin,/DESTROY
           if state.smoothingpanel then widget_control, state.panel,/DESTROY
           if state.filenamepanel then widget_control, state.panel,/DESTROY
           
           state.plotwin = widget_draw(state.plotbase,$
                                       /ALIGN_CENTER,$
                                       XSIZE=state.plot_size[0],$
                                       YSIZE=state.plot_size[1],$
                                       X_SCROLL_SIZE=state.scroll_size[0],$
                                       Y_SCROLL_SIZE=state.scroll_size[1],$
                                       /SCROLL,$
                                       EVENT_PRO='xvspec_plotwinevent',$
                                       /KEYBOARD_EVENTS,$
                                       /BUTTON_EVENTS,$
                                       /MOTION_EVENTS,$
                                       /TRACKING_EVENTS)

           if state.smoothingpanel then xvspec_addpanel,/SMOOTH
           if state.filenamepanel then xvspec_addpanel,/WRITE

           widget_control, state.plotbase,UPDATE=1           
           widget_control, state.xvspec_base,UPDATE=1
           state.mode = 'Ladder'
        
        end
        
     endcase

  endif

  
  widget_geom = widget_info(state.xvspec_base, /GEOMETRY)
  
  state.winbuffer[0]=widget_geom.xsize-state.scroll_size[0]
  state.winbuffer[1]=widget_geom.ysize-state.scroll_size[1]
  
   
  wdelete,state.pixmap_wid
  window, /FREE, /PIXMAP,XSIZE=state.plot_size[0],YSIZE=state.plot_size[1]
  state.pixmap_wid = !d.window  

end
;
;===============================================================================
;
pro xvspec_updatemenus,MODE=mode

  common xvspec_state

;  Update mode

  case state.mode of

     'Ladder': begin

        widget_control, state.but_continuous, SET_BUTTON=0
        widget_control, state.but_ladder, /SET_BUTTON
        widget_control, state.mbut_ladder, /SET_BUTTON

     end

     'Continuous': begin

        widget_control, state.but_continuous, /SET_BUTTON
        widget_control, state.but_ladder, SET_BUTTON=0
        widget_control, state.mbut_continuous, /SET_BUTTON
       
     end

  endcase

  if keyword_set(MODE) then goto, cont

  widget_control, state.xvspec_base, UPDATE=0

;  Update buffer buttons

  if state.nbuffers eq 1 then begin
  
     widget_control, state.but_buffera, /SET_BUTTON
     widget_control, state.mbut_buffera, /SET_BUTTON

  endif

  widget_control, state.but_bufferb, SENSITIVE=(state.nbuffers eq 1) ? 0:1
  widget_control, state.mbut_bufferb, SENSITIVE=(state.nbuffers eq 1) ? 0:1  
  
;  Update order numbers

  for i = 0,n_elements(*state.but_ord)-1 do begin
     
     widget_control, (*state.but_ord)[i], /DESTROY
     
  endfor
  
  *state.but_ord = lonarr(state.norders)
  for i = state.norders-1,0,-1 do begin
     
     value = strtrim(string((*state.orders)[i],FORMAT='(I3)'),2)
     
     (*state.but_ord)[i] = widget_button(state.but_order,$
                                         VALUE=value,$
                                         UVALUE='Order Menu',$
                                         FONT=state.buttonfont)
     
  endfor

;  Update aperture numbers
  
  for i = 0,n_elements(*state.but_ap)-1 do begin
     
     widget_control, (*state.but_ap)[i], /DESTROY
     
  endfor
  
  *state.but_ap = lonarr(state.naps)
  
  for i = 0,state.naps-1 do begin
     
     value = strtrim(string(i+1,FORMAT='(I2)'),2)
     (*state.but_ap)[i] = widget_button(state.but_aperture,$
                                        VALUE=value,$
                                        UVALUE='Aperture Menu',$
                                        FONT=state.buttonfont,$
                                        /CHECKED_MENU)
     
  endfor
  widget_control, (*state.but_ap)[state.ap], /SET_BUTTON

;  check Wavelength units
  
  z = where(strcompress(state.pwunits,/RE) eq strcompress(state.xunits,/RE),cnt)

  for i = 0,2 do widget_control, state.but_waveu[i], SET_BUTTON=0

  if cnt eq 1 then begin

     widget_control, state.but_wavelength,SENSITIVE=1
     *state.wunits = z
     
     widget_control, state.but_waveu[z],/SET_BUTTON
     widget_control, state.but_plotatmos,SENSITIVE=1
     widget_control, state.mbut_plotatmos,SENSITIVE=1
     widget_control, state.but_plothlines,SENSITIVE=1
     
  endif else begin

     widget_control, state.but_wavelength,SENSITIVE=0
     *state.wunits = ''

     widget_control, state.but_plotatmos,SENSITIVE=0
     widget_control, state.mbut_plotatmos,SENSITIVE=0
     widget_control, state.but_plothlines,SET_BUTTON=0,SENSITIVE=0
     widget_control, state.but_plotuserlines,SET_BUTTON=0,SENSITIVE=0     
     
  endelse
  
;  Check flux units

  z = where(strcompress(state.pfunits,/RE) eq strcompress(state.yunits,/RE),cnt)

  for i = 0,6 do widget_control, state.but_fluxu[i],SET_BUTTON=0

  if cnt eq 1 then begin
     
     widget_control, state.but_flux,SENSITIVE=1
     *state.funits = z
     widget_control, state.but_fluxu[z],/SET_BUTTON
     

  endif else begin

     widget_control, state.but_flux,SENSITIVE=0
     *state.funits = ''

  endelse

  widget_control, state.but_flags[0],SET_BUTTON=state.plotsatpixel
  widget_control, state.but_flags[1],SET_BUTTON=state.plotreplacepixel
  widget_control, state.but_flags[2],SET_BUTTON=state.plotfixpixel
  widget_control, state.but_flags[3],SET_BUTTON=state.plotoptfail
  widget_control, state.but_plotatmos,SET_BUTTON=state.plotatmosphere
  widget_control, state.mbut_plotatmos,SET_BUTTON=state.plotatmosphere

;  Check for X range

  if total(*state.xranges) eq 0 then begin

     state.plotxranges = 0
     widget_control, state.but_xranges,SET_BUTTON=state.plotxranges
     widget_control, state.but_xranges,SENSITIVE=0
     
  endif else begin

     widget_control, state.but_xranges,SENSITIVE=1
     
  endelse
  
;  Check for atmosphere

  if n_elements(*state.awave) gt 1 then begin

     widget_control, state.but_plotatmos,SENSITIVE=1
     widget_control, state.mbut_plotatmos,SENSITIVE=1

  endif else begin

     state.plotatmosphere = 0
     widget_control, state.but_plotatmos,SET_BUTTON=0
     widget_control, state.but_plotatmos,SENSITIVE=0
     widget_control, state.mbut_plotatmos,SENSITIVE=0

  endelse

;  Check for 3-color alternate
  
  if state.altcolor eq 2 then begin

     widget_control, state.but_2coloraltbut, /SET_BUTTON
     widget_control, state.mbut_2color, /SET_BUTTON
       
  endif else begin

     widget_control, state.but_3coloraltbut, /SET_BUTTON
     widget_control, state.mbut_3color, /SET_BUTTON

  endelse
     
  widget_control, state.xvspec_base, UPDATE=1
   
  cont:
    
end
;
;===============================================================================
;
pro xvspec_pickspectra,current,new

  common xvspec_state

;  Store current plot range values into proper spectype

  case current of

     'Flux': begin

        state.flxwrange = state.pwrange
        state.flxyrange = state.pyrange
        state.absflxwrange = state.pabswrange
        state.absflxyrange = state.pabsyrange

     end

     'Uncertainty': begin
        
        state.uncwrange = state.pwrange
        state.uncyrange = state.pyrange
        state.absuncwrange = state.pabswrange
        state.absuncyrange = state.pabsyrange
        
     end

     'S/N': begin

        state.snrwrange = state.pwrange
        state.snryrange = state.pyrange
        state.abssnrwrange = state.pabswrange
        state.abssnryrange = state.pabsyrange

     end

  endcase

;  Determine new spectrum and ranges
  
  case new of

     'Flux': begin

        state.pytitle = state.ytitle[0]
        
        *state.pyranges = *state.flxyranges

        if total(state.flxwrange - state.pwrange) eq 0 then begin
        
           state.pwrange = state.flxwrange
           state.pyrange = state.flxyrange

        endif else begin

;  Compute a new yrange based on the plot window wrange

           tmp = 0
           for i = 0,state.norders-1 do begin

              idx = i*state.naps+state.ap
                              
              z = where((*state.pspectra)[*,0,idx] gt state.pwrange[0] and $
                        (*state.pspectra)[*,0,idx] lt state.pwrange[1],cnt)
              if cnt ne 0 then begin

                 tmp = [tmp,(*state.pspectra)[z,1,idx]]
                 
              endif

           endfor
           tmp = tmp[1:*]
           min = min(tmp,MAX=max,/NAN)
           state.flxyrange = mc_bufrange([min,max],0.05)
           state.pyrange = state.flxyrange
           state.flxwrange = state.pwrange
           
        endelse

;  Store absolute ranges

        state.pabswrange = state.absflxwrange
        state.pabsyrange = state.absflxyrange
        
     end

     'Uncertainty': begin

        state.pytitle = state.ytitle[1]

        *state.pyranges = *state.uncyranges

        if total(state.uncwrange - state.pwrange) eq 0 then begin
           
           state.pwrange = state.uncwrange
           state.pyrange = state.uncyrange
           
        endif else begin

;  Compute a new yrange based on the plot window wrange

           tmp = 0
           for i = 0,state.norders-1 do begin

              idx = i*state.naps+state.ap              
              z = where((*state.pspectra)[*,0,idx] gt state.pwrange[0] and $
                        (*state.pspectra)[*,0,idx] lt state.pwrange[1],cnt)
              if cnt ne 0 then begin

                 tmp = [tmp,(*state.pspectra)[z,2,i]]
                 
              endif

           endfor
           tmp = tmp[1:*]
           min = min(tmp,MAX=max,/NAN)
           state.uncyrange = mc_bufrange([min,max],0.05)
           state.pyrange = state.uncyrange
           state.uncwrange = state.pwrange
          
        endelse

;  Store absolute ranges

           state.pabswrange = state.absuncwrange
           state.pabsyrange = state.absuncyrange
        
     end

     'S/N': begin

        state.pytitle = state.ytitle[2]

        *state.pyranges = *state.snryranges

        if total(state.snrwrange - state.pwrange) eq 0 then begin
        
           state.pwrange = state.snrwrange
           state.pyrange = state.snryrange

        endif else begin

;  Compute a new yrange based on the plot window wrange

           tmp = 0
           for i = 0,state.norders-1 do begin

              idx = i*state.naps+state.ap
              z = where((*state.pspectra)[*,0,idx] gt state.pwrange[0] and $
                        (*state.pspectra)[*,0,idx] lt state.pwrange[1],cnt)
              if cnt ne 0 then begin

                 tmp = [tmp,(*state.pspectra)[z,1,idx]/ $
                        (*state.pspectra)[z,2,idx]]
                 
              endif

           endfor
           tmp = tmp[1:*]
           min = min(tmp,MAX=max,/NAN)
           state.snryrange = mc_bufrange([min,max],0.05)
           state.pyrange = state.snryrange
           state.snrwrange = state.pwrange
          
        endelse

;  Store absolute ranges
        
        state.pabswrange = state.abssnrwrange
        state.pabsyrange = state.abssnryrange
        
     end

  endcase

  state.spectype = new
  
end
;
;===============================================================================
;
pro xvspec_plotspectra

  common xvspec_state

;  Update title bar
  
  widget_control, state.xvspec_base, $
                  TLB_SET_TITLE='xvspec ('+state.pbuffer+') - '+state.pfile
  
;  We plot them "backwards" so get the reverse index array
  
  idx = reverse( findgen(state.norders) * state.naps + (state.ap) )
  
;  Ladder plot
  
  if state.mode eq 'Ladder' then begin

     !p.multi[2] = state.norders
     !p.multi[0] = state.norders
          
     charsize = state.charsize
     if state.norders ge 3 then charsize = state.charsize*2.0
     xstyle = (state.plotxranges eq 1 ) ? 9:1
     
     for i = 0, state.norders-1 do begin

        j = state.norders-1-i
        
        title= '!5Order '+strtrim(string((*state.orders)[j],FORMAT='(i3)'),2)+$
               ', Aperture '+strtrim(string(state.ap+1,FORMAT='(i2)'),2)
        
        wave   = (*state.pspectra)[*,0,idx[i]]
        z = where(finite(wave) eq 1)
        wave   = wave[z]
        cflux  = (*state.pspectra)[z,1,idx[i]]
        cerror = (*state.pspectra)[z,2,idx[i]]
        flag   = (*state.pspectra)[z,3,idx[i]]
        
        case state.spectype of 
           
           'Flux': spec = cflux

           'Uncertainty': spec = cerror

           'S/N': spec = cflux/cerror
         
        endcase

        if state.plotatmosphere then begin
        
           plot,wave,spec,XSTYLE=5,/NODATA,YRANGE=[0,1],YSTYLE=5,$
                CHARSIZE=charsize,XLOG=state.xlog,BACKGROUND=20

;  Cut the spectrum to improve speed
           
           z = where(*state.awave gt (*state.wranges)[0,j] and $
                     *state.awave lt (*state.wranges)[1,j])

           oplot,(*state.awave)[z],(*state.atrans)[z],COLOR=5,PSYM=10

           !p.multi[0] = !p.multi[0]+1
           plot,wave,spec,XSTYLE=xstyle,XTITLE=state.xtitle, $
                YTITLE=state.pytitle,CHARSIZE=charsize, $
                TITLE=(state.plotxranges) ? '':title,/NODATA,$
                YRANGE=(*state.pyranges)[*,j],/YSTYLE,YLOG=state.ylog,$
                XLOG=state.xlog,BACKGROUND=20

           if state.plotxranges then axis,/XAXIS,/XSTY, $
                                          XRANGE=(*state.xranges)[*,i], $
                                          CHARSIZE=charsize,XTITLE=title
           
        endif else begin

           plot,wave,spec,XSTYLE=xstyle,XTITLE=state.xtitle, $
                YTITLE=state.pytitle,CHARSIZE=charsize, $
                TITLE=(state.plotxranges) ? '':title,/NODATA,$
                YRANGE=(*state.pyranges)[*,j],/YSTYLE,YLOG=state.ylog,$
                XLOG=state.xlog,BACKGROUND=20

           if state.plotxranges then axis,/XAXIS,/XSTY, $
                                          XRANGE=(*state.xranges)[*,i],$
                                          CHARSIZE=charsize,XTITLE=title
           
        endelse

        if state.plothlines then begin
           
;  Label H lines if requested
           
           z = where(state.hlines lt (*state.wranges)[1,j] and $
                     state.hlines gt (*state.wranges)[0,j],cnt)

           for k =0, cnt-1 do begin
              
              name = '!5'+(state.hnames)[z[k]]        
              xy1 = convert_coord((state.hlines)[z[k]],!y.crange[0], $
                                  /DATA,/TO_DEVICE)
              xy2 = convert_coord((state.hlines)[z[k]],!y.crange[1], $
                                  /DATA,/TO_DEVICE)              

              plots,[xy1[0],xy1[0]],[xy1[1],xy2[1]],LINESTYLE=1,/DEVICE, $
                    COLOR=6
              
              xyouts, xy1[0],xy2[1],name,ORIENTATION=90,/DEVICE,$
                      COLOR=6,CHARSIZE=state.charsize*0.75
              
           endfor
           
        endif

        if state.plotuserlines then begin
           
;  Label user lines if requested
           
           z = where(*state.userlines lt (*state.wranges)[1,j] and $
                     *state.userlines gt (*state.wranges)[0,j],cnt)

           for k =0, cnt-1 do begin
              
              name = '!5'+(*state.usernames)[z[k]]        
              xy1 = convert_coord((*state.userlines)[z[k]],!y.crange[0], $
                                  /DATA,/TO_DEVICE)
              xy2 = convert_coord((*state.userlines)[z[k]],!y.crange[1], $
                                  /DATA,/TO_DEVICE)              

              plots,[xy1[0],xy1[0]],[xy1[1],xy2[1]],LINESTYLE=1,/DEVICE, $
                    COLOR=7
              
              xyouts, xy1[0],xy2[1],name,ORIENTATION=90,/DEVICE,$
                      COLOR=7,CHARSIZE=state.charsize*0.75
              
           endfor
           
        endif        
        
        oplot,wave,spec,COLOR=state.color,PSYM=10
        
        if state.plotsatpixel then begin
           
           mask = mc_bitset(fix(flag),0,CANCEL=cancel)
           z = where(mask eq 1,cnt)
           plotsym,0,0.9,/FILL
           if cnt ne 0 then oplot,wave[z],spec[z],PSYM=8,COLOR=2
           
        endif
        
        
        if state.plotreplacepixel then begin
           
           mask = mc_bitset(fix(flag),1,CANCEL=cancel)
           z = where(mask eq 1,cnt)
           plotsym,0,1.0,/FILL
           if cnt ne 0 then oplot,wave[z],spec[z],PSYM=8,COLOR=4
           
        endif

        if state.plotfixpixel then begin
           
           mask = mc_bitset(fix(flag),2,CANCEL=cancel)
           z = where(mask eq 1,cnt)
           plotsym,0,1.1,/FILL
           if cnt ne 0 then oplot,wave[z],spec[z],PSYM=8,COLOR=7
           
        endif
        
        if state.plotoptfail then begin
           
           mask = mc_bitset(fix(flag),3,CANCEL=cancel)
           z = where(mask eq 1,cnt)
           plotsym,0,1.2,/FILL
           if cnt ne 0 then oplot,wave[z],spec[z],PSYM=8,COLOR=5
           
        endif
        
        if !y.crange[0] lt 0 then plots,!x.crange,[0,0],LINESTYLE=1
                
     endfor
     
     !p.multi=0


  endif

;  Continuous plot
  
  if state.mode eq 'Continuous' then begin

     position = [120,60,state.plot_size[0]-20,state.plot_size[1]-50]
     if state.norders eq 1 then position[3] = position[3]+25
     
;     if state.norders eq 1 then position[3] = position[3]+50

     noerase = 0
     ystyle = 1     
     if state.plotatmosphere then begin

        position[2] = position[2]-30     
        plot,[1],[1],XSTYLE=5,/NODATA,YRANGE=[0,1],YSTYLE=5,$
             CHARSIZE=state.charsize,XLOG=state.xlog,XRANGE=state.pwrange,$
             BACKGROUND=20,POSITION=position,/DEVICE             
        oplot,*state.awave,*state.atrans,COLOR=5,PSYM=10
        ystyle=9
        noerase = 1

     endif else noerase = 0

     plot,[1],[1],/XSTYLE,XTITLE=state.xtitle,YTITLE=state.pytitle,$
          CHARSIZE=state.charsize,TITLE=title,/NODATA,$
          YRANGE=state.pyrange,XRANGE=state.pwrange,YSTYLE=ystyle, $
          YLOG=state.ylog,XLOG=state.xlog,BACKGROUND=20,NOERASE=noerase,$
          POSITION=position,/DEVICE
     
     if state.plothlines then begin

;  Label H lines if requested
        
        z = where(state.hlines lt state.pwrange[1] and $
                  state.hlines gt state.pwrange[0],cnt)
        
        for i =0, cnt-1 do begin
           
           name = '!5'+(state.hnames)[z[i]]+'!X'        
           wave = convert_coord((state.hlines)[z[i]],1,/DATA,/TO_DEVICE)
           
           plots,[wave[0],wave[0]],[position[1],position[3]-50],LINESTYLE=1, $
                 /DEVICE,COLOR=6
           
           xyouts, wave[0],position[3]-50,name,ORIENTATION=90,/DEVICE,COLOR=6,$
                   CHARSIZE=state.charsize*0.75
           
        endfor

     endif

;  Label user lines if requested
     
     if state.plotuserlines then begin
        
        z = where(*state.userlines lt state.pwrange[1] and $
                  *state.userlines gt state.pwrange[0],cnt)
        
        for i =0, cnt-1 do begin
           
           name = '!5'+(*state.usernames)[z[i]]+'!X'        
           wave = convert_coord((*state.userlines)[z[i]],1,/DATA,/TO_DEVICE)
           
           plots,[wave[0],wave[0]],[position[1],position[3]-50], $
                 LINESTYLE=1,/DEVICE,COLOR=7
           
           xyouts, wave[0],position[3]-50,name,ORIENTATION=90,/DEVICE, $
                   COLOR=7,CHARSIZE=state.charsize*0.75
           
        endfor

     endif
           
     if state.norders gt 1 then $
        xyouts,20,state.plot_size[1]-30,'!5Order',ALIGNMENT=0,/DEVICE, $
               CHARSIZE=state.charsize

     for i = 0, state.norders-1 do begin

        wave   = (*state.pspectra)[*,0,i*state.naps+state.ap]
        z      = where(finite(wave) eq 1)
        wave   = wave[z]

;  Only plot orders in the wavelength range
        
        min = min(wave,MAX=max)
        if min gt (*state.wranges)[1,i] or $
           max lt (*state.wranges)[0,i] then continue

        cflux  = (*state.pspectra)[z,1,i*state.naps+state.ap]
        cerror = (*state.pspectra)[z,2,i*state.naps+state.ap]
        flag   = (*state.pspectra)[z,3,i*state.naps+state.ap]
        
        case state.spectype of 
           
           'Flux': spec = cflux
           
           'Uncertainty': spec = cerror
           
           'S/N': spec = cflux/cerror

        endcase

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
        
        oplot,wave,spec,COLOR=color,PSYM=10

;  Label order numbers

        if state.norders gt 1 then begin
        
           min = max([!x.crange[0],(*state.wranges)[0,i]])
           max = min([!x.crange[1],(*state.wranges)[1,i]])
           
           lwave = (min+max)/2.
           
           if lwave gt !x.crange[0] then begin
              
              xy = convert_coord(lwave,!y.crange[1],/DATA,/TO_DEVICE)
              
              xyouts,xy[0],xy[1]+10+15*(i mod 2), $
                     strtrim(string((*state.orders)[i],FORMAT='(I3)'),2), $
                     /DEVICE,COLOR=color,ALIGNMENT=0.5,CHARSIZE=state.charsize


           endif
           
        endif

;  Add various flags
        
        if state.plotsatpixel then begin

           mask = mc_bitset(fix(flag),0,CANCEL=cancel)
           if cancel then return

           z = where(mask eq 1,cnt)
           if cnt ne 0 then begin

              plotsym,0,0.9,/FILL
              oplot,wave[z],spec[z],PSYM=8,COLOR=2

           endif
           
        endif
                
        if state.plotreplacepixel then begin
           
           mask = mc_bitset(fix(flag),1,CANCEL=cancel)
           z = where(mask eq 1,cnt)
           if cnt ne 0 then begin

              plotsym,0,1.0,/FILL
              oplot,wave[z],spec[z],PSYM=8,COLOR=4

           endif
              
        endif

        if state.plotfixpixel then begin
           
           mask = mc_bitset(fix(flag),2,CANCEL=cancel)
           z = where(mask eq 1,cnt)
           if cnt ne 0 then begin

              plotsym,0,1.1,/FILL
              oplot,wave[z],spec[z],PSYM=8,COLOR=7

           endif
              
        endif
        
        if state.plotoptfail then begin
           
           mask = mc_bitset(fix(flag),3,CANCEL=cancel)
           z = where(mask eq 1,cnt)
           if cnt ne 0 then begin

              plotsym,0,1.2,/FILL
              oplot,wave[z],spec[z],PSYM=8,COLOR=5

           endif
           
        endif
        
     endfor

     if ystyle eq 9 then begin
        
        ticks = ['0.0','0.2','0.4','0.6','0.8','1.0']
        axis,YAXIS=1,YTICKS=5,YTICKNAME=ticks,YMINOR=2,COLOR=5, $
             CHARSIZE=state.charsize
        
     endif
     
     if !y.crange[0] lt 0 then plots,!x.crange,[0,0],LINESTYLE=1

     state.xscale = !x
     state.yscale = !y
     state.pscale = !p
     
  endif
  
end
;
;===============================================================================
;
pro xvspec_plotupdate

  common xvspec_state

  mc_mkct
  wset, state.pixmap_wid
  erase,COLOR=20
  xvspec_plotspectra
  
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0, $
                state.pixmap_wid]

  xvspec_setminmax
  
end
;
;===============================================================================
;
pro xvspec_setminmax

  common xvspec_state
  
  if state.mode eq 'Ladder' then return
  
  widget_control, state.xmin_fld[1],SET_VALUE=strtrim(state.pwrange[0],2)
  
  widget_control, state.xmax_fld[1],SET_VALUE=strtrim(state.pwrange[1],2)

  widget_control, state.ymin_fld[1],SET_VALUE=strtrim(state.pyrange[0],2)

  widget_control, state.ymax_fld[1],SET_VALUE=strtrim(state.pyrange[1],2)

  xvspec_setslider
  
end
;
;=============================================================================
;
pro xvspec_setslider

  common xvspec_state

;  Get new slider value
  
  del = state.pabswrange[1]-state.pabswrange[0]
  midwave = (state.pwrange[1]+state.pwrange[0])/2.
  state.sliderval = round((midwave-state.pabswrange[0])/del*100)
  
  widget_control, state.slider, SET_VALUE=state.sliderval
  
end
;
;===============================================================================
;
pro xvspec_snrcut

  common xvspec_state

  if state.snrcut eq 0 then begin

     *state.amspectra = *state.aspectra
     *state.bmspectra = *state.bspectra     

  endif else begin

     for i = 0,state.norders-1 do begin
        
        for j = 0,state.naps-1 do begin

           y = (*state.aspectra)[*,1,i*state.naps+j]
           e = (*state.aspectra)[*,2,i*state.naps+j]

           z = where(y/e lt state.snrcut,cnt)
           if cnt ne 0 then begin

              y[z] = !values.f_nan
              (*state.amspectra)[*,1,i*state.naps+j] = y
              
           endif
           
           if state.nbuffers eq 2 then begin

              y = (*state.bspectra)[*,1,i*state.naps+j]
              e = (*state.bspectra)[*,2,i*state.naps+j]
              
              z = where(y/e lt state.snrcut,cnt)
              if cnt ne 0 then begin
                 
                 y[z] = !values.f_nan
                 (*state.bmspectra)[*,1,i*state.naps+j] = y
                 
              endif

           endif
           
        endfor
        
     endfor
 
  endelse

  case state.buffer of
     
     0: *state.pspectra = *state.amspectra
     1: *state.pspectra = *state.bmspectra
     
  endcase
  
end
;
;===============================================================================
;
pro xvspec_smoothspectra

  common xvspec_state

  if state.fwhm eq 0 then begin

     *state.amspectra = *state.aspectra
     *state.bmspectra = *state.bspectra
     
  endif else begin

     for i = 0,state.norders-1 do begin
        
        for j = 0,state.naps-1 do begin

           y = (*state.aspectra)[*,1,i*state.naps+j]
           e = (*state.aspectra)[*,2,i*state.naps+j]
           x = findgen(n_elements(y))

           mc_convolvespec,x,y,state.fwhm,ny,oe,ERROR=e,CANCEL=cancel
           if cancel then return

           (*state.amspectra)[*,1,i*state.naps+j] = ny
           (*state.amspectra)[*,2,i*state.naps+j] = oe

           if state.nbuffers eq 2 then begin

              y = (*state.bspectra)[*,1,i*state.naps+j]
              e = (*state.bspectra)[*,2,i*state.naps+j]
              x = findgen(n_elements(y))
              
              mc_convolvespec,x,y,state.fwhm,ny,oe,ERROR=e,CANCEL=cancel
              if cancel then return
              
              (*state.bmspectra)[*,1,i*state.naps+j] = ny
              (*state.bmspectra)[*,2,i*state.naps+j] = oe             

           endif
           
        endfor
        
     endfor
     
  endelse 

  case state.buffer of
     
     0: *state.pspectra = *state.amspectra
     1: *state.pspectra = *state.bmspectra
     
  endcase
  
end
;
; *****************************************************************************
;
pro xvspec_writefile,FITS=fits,ASCII=ascii

  common xvspec_state

  filename = mc_cfld(state.tmp_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
;  Get hdr info

  case state.buffer of

     0: hdr = *state.ahdr

     1: hdr = *state.bhdr
  
  endcase

  if *state.funits ne '' then begin
  
     ytitle = mc_getfunits((*state.funits)[0],UNITS=yunits,CANCEL=cancel)
     if cancel then return

     fxaddpar,hdr,'YUNITS',yunits
     fxaddpar,hdr,'YTITLE',ytitle

  endif

  if *state.wunits ne '' then begin
  
     mc_getwunits,(*state.wunits)[0],xunits,xtitle,CANCEL=cancel
     if cancel then return
     
     fxaddpar,hdr,'XUNITS',xunits
     fxaddpar,hdr,'XTITLE',xtitle

  endif

  case state.buffer of

     0: pspectra = *state.amspectra
     1: pspectra = *state.bmspectra

  endcase
  
  if keyword_set(FITS) then begin
     
     writefits,state.path+filename+'.fits',pspectra,hdr

  endif

  if keyword_set(ASCII) then begin

     npix = long(fxpar(hdr,'NAXIS1'))
     
     openw,lun,state.path+filename+'.txt', /GET_LUN
     
     for j = 0, n_elements(hdr)-1 do printf, lun, '#'+hdr[j]
     
     for j = 0L, npix[0]-1L do begin
        
        printf, lun,  strjoin( reform((pspectra)[j,*,*], $
                                      4*state.naps*state.norders),'  ' )
        
     endfor

     free_lun, lun     

  endif

  state.filenamepanel = 0
  
  widget_control, state.panel, /DESTROY
  
end
;
;=============================================================================
;
pro xvspec_zoom,IN=in,OUT=out

  common xvspec_state

  delabsx = state.pabswrange[1]-state.pabswrange[0]
  delx    = state.pwrange[1]-state.pwrange[0]
  
  delabsy = state.pabsyrange[1]-state.pabsyrange[0]
  dely    = state.pyrange[1]-state.pyrange[0]
  
  xcen = state.pwrange[0]+delx/2.
  ycen = state.pyrange[0]+dely/2.
  
  case state.cursormode of 
     
     'XZoom': begin
        
        z = alog10(delabsx/delx)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsx/2.^z/2.
        state.pwrange = [xcen-hwin,xcen+hwin]
        
     end
     
     'YZoom': begin
        
        z = alog10(delabsy/dely)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsy/2.^z/2.
        state.pyrange = [ycen-hwin,ycen+hwin]
        
     end
     
     'Zoom': begin
        
        z = alog10(delabsx/delx)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsx/2.^z/2.
        state.pwrange = [xcen-hwin,xcen+hwin]
        
        z = alog10(delabsy/dely)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsy/2.^z/2.
        state.pyrange = [ycen-hwin,ycen+hwin]
        
     end
     
     else:
     
endcase

  xvspec_plotupdate
  
end
;
;===============================================================================
;
; ------------------------------Event Handlers-------------------------------- 
;
;===============================================================================
;
pro xvspec_event,event

  common xvspec_state

  widget_control, event.id,  GET_UVALUE=uvalue
  widget_control, /HOURGLASS

  case uvalue of

     'Aperture Menu': begin

        for i = 0,state.naps-1 do widget_control, (*state.but_ap)[i], $
                                                  SET_BUTTON=0

        z = where(*state.but_ap eq event.id)
        widget_control, (*state.but_ap)[z],/SET_BUTTON
        state.ap = z[0]
        xvspec_plotupdate

     end

     'Buffer A': begin
        
        widget_control, state.but_buffera,/SET_BUTTON
        widget_control, state.but_bufferb,SET_BUTTON=0
        widget_control, state.mbut_buffera,/SET_BUTTON
        widget_control, state.mbut_bufferb,SET_BUTTON=0
        state.buffer=0
        *state.pspectra = *state.amspectra
        state.pbuffer = 'A'
        state.pfile = state.afile
        xvspec_plotupdate
        
     end

     'Buffer B': begin

        widget_control, state.but_bufferb,/SET_BUTTON
        widget_control, state.but_buffera,SET_BUTTON=0
        widget_control, state.mbut_bufferb,/SET_BUTTON
        widget_control, state.mbut_buffera,SET_BUTTON=0
        state.buffer=1
        *state.pspectra = *state.bmspectra
        state.pbuffer = 'B'
        state.pfile = state.bfile
        xvspec_plotupdate
        
     end
     
     'Continuous Plot Button': begin

        widget_control, state.but_ladder,SET_BUTTON=0
        widget_control, state.but_continuous,SET_BUTTON=1
        widget_control, state.mbut_continuous, /SET_BUTTON
        xvspec_modwin,/CONTINUOUS
        xvspec_plotupdate

     end

     'Done Smoothing Button': begin
        
        state.smoothingpanel = 0

        widget_control, state.but_smooth,SET_BUTTON=0
        widget_control, state.panel, /DESTROY

     end

     'Done S/N Cut Button': begin
        
        state.snrcutpanel = 0

        widget_control, state.but_snrcut,SET_BUTTON=0
        widget_control, state.panel, /DESTROY

     end
     
     'Close Write Button': begin
        
        state.filenamepanel = 0
        widget_control, state.panel, /DESTROY

     end

     
     'Gaussian FWHM': begin

        fwhm = mc_cfld(state.tmp_fld,3,/EMPTY,CANCEL=cancel)
        if cancel then return
        
        state.fwhm = fwhm
        xvspec_smoothspectra
        xvspec_plotupdate

     end

     'Fixed Pixel Menu': begin

        state.plotfixpixel = ~state.plotfixpixel
        widget_control, state.but_flags[2],SET_BUTTON=state.plotfixpixel
        xvspec_plotupdate
        
     end

     'Fix Y Range Button': begin

        state.fix = (state.fix eq 1) ? 0:1
        widget_control, state.but_fixyrange,SET_BUTTON=state.fix
        xvspec_plotupdate

     end

     'Flux Units Menu': begin

        for i = 0,6 do widget_control, state.but_fluxu[i],SET_BUTTON=0

        z = where(state.but_fluxu eq event.id)
        widget_control, state.but_fluxu[z],/SET_BUTTON

        *state.funits = [*state.funits,z]
        xvspec_convertflux
        xvspec_plotupdate

     end

     'Help Button': begin

        xmc_displaytext,filepath('xvspec_helpfile.txt', $
                                 ROOT_DIR=state.spextoolpath, $
                                 SUBDIR='helpfiles'), $
                        TITLE='xvspec Help File', $
                        GROUP_LEADER=state.xvspec_base, $
                        WSIZE=[600,400]

     end

     'Ladder Plot Button': begin

        if state.norders eq 1 then begin

           ok = dialog_message('Only one order so ladder mode is pointless.', $
                               /INFO, DIALOG_PARENT=state.xvspec_base)
           
           widget_control, state.mbut_continuous,SET_BUTTON=1
           return

        endif
          
        widget_control, state.but_ladder,SET_BUTTON=1
        widget_control, state.but_continuous,SET_BUTTON=0
        widget_control, state.mbut_ladder, /SET_BUTTON
        xvspec_modwin,/LADDER
        xvspec_plotupdate
        
     end

     'Load Spextool FITS Button': begin

        fullpath = dialog_pickfile(DIALOG_PARENT=state.xvspec_base,$
                                   FILTER='*.fits', $
                                   PATH=state.path,GET_PATH=newpath, $
                                   /MUST_EXIST)
        state.path = newpath

        if fullpath eq '' then goto, cont else begin
           
           xvspec_loadspectra,fullpath
           xvspec_modwin
           xvspec_updatemenus

           *state.pspectra = *state.amspectra
           state.pbuffer = 'A'
           state.pfile = state.afile
           
;  Get the ranges

           xvspec_getranges

;  Set the view to be flux
           
           state.spectype = 'Flux'
           widget_control, state.mbut_flx,/SET_BUTTON
           widget_control, state.but_flx,/SET_BUTTON
           widget_control, state.but_unc,SET_BUTTON=0
           widget_control, state.but_snr,SET_BUTTON=0
           
           state.pytitle = state.ytitle[0]
           
           *state.pyranges = *state.flxyranges
           
           state.pwrange = state.flxwrange
           state.pabswrange = state.absflxwrange
           
           state.pyrange = state.flxyrange
           state.pabsyrange = state.absflxyrange
           
           xvspec_plotupdate
           
        endelse
        
     end

     'Load User Lines Button': begin

        fullpath = dialog_pickfile(DIALOG_PARENT=state.xvspec_base,$
                                   /MUST_EXIST)
        
        if fullpath ne '' then begin

           readcol,fullpath,x,y,DELIMITER='|',COMMENT='#',FORMAT='D,A', $
                   /SILENT
           *state.userlines = x
           *state.usernames = y
           state.plotuserlines = 1
           widget_control, state.but_plotuserlines,SENSITIVE=1,/SET_BUTTON     
           xvspec_plotupdate
           
        endif
        
     end
     
     'Order Menu': begin

        if state.mode eq 'Ladder' then begin
        
           if state.plot_size[1] eq state.scroll_size[1] then goto, cont
           
           for i = 0,state.norders-1 do widget_control, (*state.but_ord)[i], $
              SET_BUTTON=0
           
           z = where(*state.but_ord eq event.id)
           
           del = max(*state.orders,MIN=min)-min+1
           offset = state.plot_size[1]/float((del+1))
           frac = ((*state.orders)[z]-min)/float(del)
           
           widget_control, state.plotwin, $
                           SET_DRAW_VIEW=[0,state.plot_size[1]*frac-offset]
           
        end

        if state.mode eq 'Continuous' then begin

           z = where(*state.but_ord eq event.id)

           oldcen = (state.pwrange[1]+state.pwrange[0])/2.
           newcen = mean((*state.wranges)[*,z])

           state.pwrange = state.pwrange + (newcen-oldcen)
           xvspec_plotupdate
           
        end
           
     end

     'Opt Extract Fail Menu': begin

        state.plotoptfail = ~state.plotoptfail
        widget_control, state.but_flags[3],SET_BUTTON=state.plotoptfail
        xvspec_plotupdate
        
     end
    
     'Plot Atmosphere Button': begin

        state.plotatmosphere = ~state.plotatmosphere
        widget_control, state.but_plotatmos, SET_BUTTON=state.plotatmosphere
        widget_control, state.mbut_plotatmos, SET_BUTTON=state.plotatmosphere
        xvspec_plotupdate
        
     end

     'Plot Flux Button': begin

        widget_control, state.but_flx,SET_BUTTON=1
        widget_control, state.but_unc,SET_BUTTON=0
        widget_control, state.but_snr,SET_BUTTON=0
        widget_control, state.mbut_flx,/SET_BUTTON
        xvspec_pickspectra,state.spectype,'Flux'
        xvspec_plotupdate

     end

     'Plot Hydrogen Lines Button': begin

        state.plothlines = ~state.plothlines
        widget_control, state.but_plothlines, SET_BUTTON=state.plothlines
        xvspec_plotupdate
        
     end
     
     'Plot S/N Button': begin

        widget_control, state.but_flx,SET_BUTTON=0
        widget_control, state.but_unc,SET_BUTTON=0
        widget_control, state.but_snr,SET_BUTTON=1
        widget_control, state.mbut_snr,/SET_BUTTON
        xvspec_pickspectra,state.spectype,'S/N'
        xvspec_plotupdate

     end

     'Plot Uncertainty Button': begin

        widget_control, state.but_flx,SET_BUTTON=0
        widget_control, state.but_unc,SET_BUTTON=1
        widget_control, state.but_snr,SET_BUTTON=0
        widget_control, state.mbut_unc,/SET_BUTTON
        xvspec_pickspectra,state.spectype,'Uncertainty'
        xvspec_plotupdate

     end

     'Plot User Lines Button': begin

        state.plotuserlines = ~state.plotuserlines
        widget_control, state.but_plotuserlines, SET_BUTTON=state.plotuserlines
        xvspec_plotupdate
        
     end
     
     'Replaced Pixel Menu': begin

        state.plotreplacepixel = ~state.plotreplacepixel
        widget_control, state.but_flags[1],SET_BUTTON=state.plotreplacepixel
        xvspec_plotupdate
        
     end
     
     'Saturated Pixel Menu': begin

        state.plotsatpixel = ~state.plotsatpixel
        widget_control, state.but_flags[0],SET_BUTTON=state.plotsatpixel
        xvspec_plotupdate
        
     end
     
     'Slider': begin

           del = state.pabswrange[1]-state.pabswrange[0]
           oldcen = (state.pwrange[1]+state.pwrange[0])/2.
           newcen = state.pabswrange[0]+del*(event.value/100.)
           
           state.pwrange = state.pwrange + (newcen-oldcen)
           xvspec_plotupdate
                      
     end

     'Smooth Button': begin

        if state.smoothingpanel then return

        state.smoothingpanel = 1
        widget_control, state.xvspec_base,UPDATE=0
        xvspec_addpanel,/SMOOTH
        widget_control, state.xvspec_base,UPDATE=1
        
     end

     'S/N Cut Button': begin

        if state.snrcutpanel then return
        
        state.snrcutpanel = 1
        widget_control, state.xvspec_base,UPDATE=0
        xvspec_addpanel,/SNRCUT
        widget_control, state.xvspec_base,UPDATE=1
        
     end

     'S/N Cut': begin

        cut = mc_cfld(state.tmp_fld,3,/EMPTY,CANCEL=cancel)
        if cancel then return
        state.snrcut = cut
        xvspec_snrcut
        xvspec_plotupdate

     end
     
     'Quit': widget_control, event.top, /DESTROY

     'View FITS Header Button': begin
        
        case state.buffer of
           
           0: hdr = *state.ahdr
           
           1: hdr = *state.bhdr
           
        endcase
                
        if n_elements(hdr) le 1 then begin
           
           ok = dialog_message('No FITS header.',/INFO, $
                               DIALOG_PARENT=state.xvspec_base)
           
           
        endif else begin
           
           xmc_displaytext,hdr,TITLE='FITS Header', $
                           GROUP_LEADER=state.xvspec_base
           
        endelse
        
     end

     'Wavelength Units Menu': begin

        for i = 0,2 do widget_control, state.but_waveu[i],SET_BUTTON=0

        z = where(state.but_waveu eq event.id)
        widget_control, state.but_waveu[z],/SET_BUTTON

        *state.wunits = [*state.wunits,z]
        xvspec_convertwave
        xvspec_plotupdate

     end

     'Write ASCII File': begin

        if state.filenamepanel then return
        
        state.filenamepanel = 1
        
        widget_control, state.xvspec_base,UPDATE=0        
        xvspec_addpanel,/ASCII
        widget_control, state.xvspec_base,UPDATE=1
        
     end

     'Write ASCII Filename': xvspec_writefile,/ASCII

     'Write Spextool FITS File': begin

        if state.filenamepanel then return

        widget_control, state.xvspec_base,UPDATE=0        
        xvspec_addpanel,/FITS
        widget_control, state.xvspec_base,UPDATE=1
        
     end

     'Write FITS Filename': xvspec_writefile,/FITS
             
     'X Log Button': begin

        state.xlog = (state.xlog eq 1) ? 0:1
        widget_control, state.but_xlog,SET_BUTTON=state.xlog
        xvspec_plotupdate

     end

     'X Ranges Button': begin

        if total(*state.xranges) eq 0 then return
        state.plotxranges = (state.plotxranges eq 1) ? 0:1
        widget_control, state.but_xranges,SET_BUTTON=state.plotxranges
        xvspec_plotupdate
        
     end
     
     'Y Log Button': begin

        state.ylog = (state.ylog eq 1) ? 0:1
        widget_control, state.but_ylog,SET_BUTTON=state.ylog
        xvspec_plotupdate

     end

     '2-Color Alternate Spectra Button': begin

        widget_control, state.but_3coloraltbut, SET_BUTTON=0
        widget_control, state.but_2coloraltbut, SET_BUTTON=1
        widget_control, state.mbut_2color, /SET_BUTTON
        state.altcolor = 2
        xvspec_plotupdate
        
     end

     '3-Color Alternate Spectra Button': begin

        widget_control, state.but_3coloraltbut, SET_BUTTON=1
        widget_control, state.but_2coloraltbut, SET_BUTTON=0
        widget_control, state.mbut_3color, /SET_BUTTON
        state.altcolor = 3
        xvspec_plotupdate
        
     end

     else:
     
  endcase
  
cont:
  
end
;
;===============================================================================
;
pro xvspec_minmaxevent,event

  common xvspec_state
  
  widget_control, event.id,  GET_UVALUE = uvalue
  
  case uvalue of 
     
     'X Min': begin
        
        xmin = mc_cfld(state.xmin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        xmin2 = mc_crange(xmin,state.pwrange[1],'X Min',/KLT,$
                          WIDGET_ID=state.xvspec_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control, state.xmin_fld[0],SET_VALUE=state.pwrange[0]
           return
           
        endif else state.pwrange[0] = xmin2
        
     end

     'X Max': begin
        
        xmax = mc_cfld(state.xmax_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        xmax2 = mc_crange(xmax,state.pwrange[0],'X Max',/KGT,$
                       WIDGET_ID=state.xvspec_base,CANCEL=cancel)
        if cancel then begin
            
            widget_control, state.xmax_fld[0],SET_VALUE=state.pwrange[1]
            return
            
         endif else state.pwrange[1] = xmax2
        
     end
     
     'Y Min': begin
        
        ymin = mc_cfld(state.ymin_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        ymin2 = mc_crange(ymin,state.pyrange[1],'Y Min',/KLT,$
                          WIDGET_ID=state.xvspec_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control,state.ymin_fld[0],SET_VALUE=state.pyrange[0]
            return
            
        endif else state.pyrange[0] = ymin2
        
     end
     
     'Y Max': begin

        ymax = mc_cfld(state.ymax_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        ymax2 = mc_crange(ymax,state.pyrange[0],'Y Max',/KGT,$
                          WIDGET_ID=state.xvspec_base,CANCEL=cancel)
        if cancel then begin
           
           widget_control,state.ymax_fld[0],SET_VALUE=state.pyrange[1]
           return
           
        endif else state.pyrange[1] = ymax2
        
     end
     
  endcase
  
  xvspec_plotupdate
  
end
;
;===============================================================================
;
pro xvspec_plotwinevent,event

  common xvspec_state

  widget_control, event.id,  GET_UVALUE=uvalue

;  Check tracking
  
  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin

     wset, state.plotwin_wid
     device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0, $
                   state.pixmap_wid]
     widget_control, state.plotwin,INPUT_FOCUS=event.enter

     return

  endif

;  Check for ladder events
  
  if state.mode eq 'Ladder' then begin

     if event.type eq 6 and event.press eq 1 then begin
        
        case event.key of

           7: begin             ;  up arrow
              
              widget_control, state.plotwin,GET_DRAW_VIEW=current
              offset = state.plot_size[1]/state.norders
              max = state.plot_size[1]-state.scroll_size[1]
              val = (current[1]+offset) < max
              widget_control, state.plotwin,SET_DRAW_VIEW=[0,val]
              return
              
           end
           
           8: begin             ;  down arrow
              
              widget_control, state.plotwin,GET_DRAW_VIEW=current
              offset = state.plot_size[1]/state.norders
              max = state.plot_size[1]-state.scroll_size[1]
              val = (current[1]-offset) > 0
              widget_control, state.plotwin,SET_DRAW_VIEW=[0,val]
              return
              
           end
           
           else:
           
        endcase

     endif

;  Now check for the xzoomplot call
     
     if event.type eq 1 then begin

        idx = floor(event.y/float(state.plot_size[1])*state.norders)
        sidx = idx * state.naps + (state.ap)
        
        case state.buffer of
           
           0: begin
              
              cwave  = (*state.amspectra)[*,0,sidx]
              cflux  = (*state.amspectra)[*,1,sidx]
              cerror = (*state.amspectra)[*,2,sidx]
              
           end
           
           1: begin
              
              cwave  = (*state.bmspectra)[*,0,sidx]
              cflux  = (*state.bmspectra)[*,1,sidx]
              cerror = (*state.bmspectra)[*,2,sidx]
              
           end
           
        endcase
        
        case state.spectype of 
           
           'Flux': begin
              
              spec   = cflux
              xtitle = state.xtitle
              ytitle = state.ytitle[0]
              
           end
           
           'Uncertainty': begin
              
              spec   = cerror
              xtitle = state.xtitle
              ytitle = state.ytitle[1]
              
           end
           
           'S/N': begin
              
              spec   = cflux/cerror
              xtitle = state.xtitle
              ytitle = '!5S/N'
              
           end
           
        endcase
        
        title= '!5Order '+ $
               strtrim(string((*state.orders)[idx],FORMAT='(i3)'),2)+$
               ', Aperture '+strtrim(string(state.ap+1,FORMAT='(i2)'),2)
        xzoomplot,cwave,spec,XTITLE=xtitle,YTITLE=ytitle,YLOG=state.ylog, $
                  TITLE=title
        
     endif

;
;  Check for ASCII keyboard event
;
     if event.type eq 5 and event.release eq 1 then begin
  
        case strtrim(event.ch,2) of 
           
           'q': widget_control, event.top, /DESTROY
           
           else:
           
        endcase
        return
        
     endif
     
  endif

  if state.mode eq 'Continuous' then begin

;  Check for arrow keys
     
     if event.type eq 6 and event.release eq 1 then begin
        
        case event.modifiers of
           
           0: begin
              
              case event.key of
                 
                 5: begin       ; left arrow

                    del = (state.pwrange[1]-state.pwrange[0])*0.3
                    oldcen = (state.pwrange[1]+state.pwrange[0])/2.
                    newcen = oldcen-del

                    if newcen lt state.pabswrange[0] then return
                    state.pwrange = state.pwrange + (newcen-oldcen)
                    xvspec_plotupdate
                    
                 end
                 
                 6: begin       ; right arrow
                    
                    del = (state.pwrange[1]-state.pwrange[0])*0.3
                    oldcen = (state.pwrange[1]+state.pwrange[0])/2.
                    newcen = oldcen+del

                    if newcen gt state.pabswrange[1] then return
                    state.pwrange = state.pwrange + (newcen-oldcen)
                    xvspec_plotupdate
                    
                 end
                 
                 7: begin       ;  up arrow
                    
                    widget_control, state.plotwin,GET_DRAW_VIEW=current
                    offset = state.plot_size[1]/state.norders
                    max = state.plot_size[1]-state.scroll_size[1]
                    val = (current[1]+offset) < max
                    widget_control, state.plotwin,SET_DRAW_VIEW=[0,val]
                    return
                    
                 end
                 
                 8: begin       ;  down arrow
                    
                    widget_control, state.plotwin,GET_DRAW_VIEW=current
                    offset = state.plot_size[1]/state.norders
                    max = state.plot_size[1]-state.scroll_size[1]
                    val = (current[1]-offset) > 0
                    widget_control, state.plotwin,SET_DRAW_VIEW=[0,val]
                    return
                    
                 end
                 
                 else:
                 
              endcase
              
           end
           
           1: begin
              
              case event.key of
                 
                 5: tvcrs,event.x-1,event.y,/DEVICE
                 
                 6: tvcrs,event.x+1,event.y,/DEVICE                 
                 
                 else:
                 
              endcase
              
           end
           
           else:
           
        endcase
        
     endif
;
;  Check for ASCII keyboard event
;
     if event.type eq 5 and event.release eq 1 then begin
  
        case strtrim(event.ch,2) of 
           
           'a': begin
              
              state.pabswrange = state.pwrange
              state.pabsyrange= state.pyrange
              
           end
        
           'c': begin          
              
              state.cursormode = 'None'
              state.reg = !values.f_nan                
              xvspec_plotupdate
              
           end
           
           'i': xvspec_zoom,/IN
           
           'o': xvspec_zoom,/OUT
           
           'q': widget_control, event.top, /DESTROY
           
           'm': begin 
              
              if state.mode eq 'Continuous' then begin
                 
                 !p = state.pscale
                 !x = state.xscale
                 !y = state.yscale
                 x  = event.x/float(state.plot_size[0])
                 y  = event.y/float(state.plot_size[1])
                 xy = convert_coord(x,y,/NORMAL,/TO_DATA,/DOUBLE)
                 
                 print, xy[0:1]
                 
              endif
              
           end
           
           'w': begin
              
              state.pwrange = state.pabswrange
              state.pyrange = state.pabsyrange
              xvspec_plotupdate
              
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

     wset, state.plotwin_wid
     
     !p = state.pscale
     !x = state.xscale
     !y = state.yscale
     x  = event.x/float(state.plot_size[0])
     y  = event.y/float(state.plot_size[1])
     xy = convert_coord(x,y,/NORMAL,/TO_DATA,/DOUBLE)
     
     if event.type eq 1 then begin
     
        if state.cursormode eq 'None' then return
        z = where(finite(state.reg) eq 1,count)
        if count eq 0 then begin
           
           wset, state.pixmap_wid
           state.reg[*,0] = xy[0:1]
           case state.cursormode of
              
              'XZoom': plots, [event.x,event.x],[0,state.plot_size[1]], $
                              COLOR=2,/DEVICE,LINESTYLE=2
              
              'YZoom': plots, [0,state.plot_size[0]],[event.y,event.y], $
                              COLOR=2,/DEVICE,LINESTYLE=2
              
              else:
              
           endcase
           wset, state.plotwin_wid
           device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0, $
                         state.pixmap_wid]
           
        endif else begin 
           
           state.reg[*,1] = xy[0:1]
           case state.cursormode of 
              
              'XZoom': state.pwrange = [min(state.reg[0,*],MAX=max),max]
              
              'YZoom': state.pyrange = [min(state.reg[1,*],MAX=max),max]
              
              'Zoom': begin
                 
                 state.pwrange = [min(state.reg[0,*],MAX=max),max]
                 state.pyrange = [min(state.reg[1,*],MAX=max),max]
                 
              end
              
           endcase
           xvspec_plotupdate
           state.cursormode='None'
           
        endelse

     endif

  endif

;  Copy the pixmaps and draw the cross hair or zoom lines.
     
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0, $
                state.pixmap_wid]
  
  case state.cursormode of 
     
     'XZoom': plots, [event.x,event.x],[0,state.plot_size[1]], $
                     COLOR=2,/DEVICE
     
     'YZoom': plots, [0,state.plot_size[0]],[event.y,event.y], $
                     COLOR=2,/DEVICE
     
     'Zoom': begin
        
        plots, [event.x,event.x],[0,state.plot_size[1]],COLOR=2,/DEVICE
        plots, [0,state.plot_size[0]],[event.y,event.y],COLOR=2,/DEVICE
        xy = convert_coord(event.x,event.y,/DEVICE,/TO_DATA,/DOUBLE)
        plots,[state.reg[0,0],state.reg[0,0]],[state.reg[1,0],xy[1]],$
              LINESTYLE=2,COLOR=2
        plots,[state.reg[0,0],xy[0]],[state.reg[1,0],state.reg[1,0]],$
              LINESTYLE=2,COLOR=2
        
     end
     
     else: begin
        
        plots, [event.x,event.x],[0,state.plot_size[1]],COLOR=2,/DEVICE
        plots, [0,state.plot_size[0]],[event.y,event.y],COLOR=2,/DEVICE
        
     end
     
  endcase
    
;  Update the cursor information
  
  beam = (state.nbuffers eq 1) ? 'A':'A,B'
  label = 'Orders: '+strtrim(string(state.norders,FORMAT='(I2)'),2)+$
          ', Aps: '+strtrim(string(state.naps,FORMAT='(I2)'),2)+$
          ', Beams: '+beam
  
  if state.mode eq 'Continuous' then begin
     
     label = label+', Cursor (X,Y): '+strtrim(xy[0],2)+', '+strtrim(xy[1],2)

  endif
  widget_control,state.message,SET_VALUE=label

end
;
;===============================================================================
;
pro xvspec_resizeevent, event

  common xvspec_state

  widget_control, state.xvspec_base, TLB_GET_SIZE=size

  xsize = size[0]
  ysize = size[1]
  
  case state.mode of
     
     'Continuous': begin
        
        state.plot_size[1] = ysize-state.winbuffer[1]
        state.plot_size[0] = xsize-state.winbuffer[0]
        
        widget_control, state.plotwin, DRAW_XSIZE=state.plot_size[0], $
                        DRAW_YSIZE=state.plot_size[1]           
        
        widget_geom = widget_info(state.xvspec_base, /GEOMETRY)
        
        state.winbuffer[0]=widget_geom.xsize-state.plot_size[0]
        state.winbuffer[1]=widget_geom.ysize-state.plot_size[1]
        
     end
     
     'Ladder': begin
        
;  Figure out new size           
        
        state.plot_size[0] = xsize-state.winbuffer[0]
        state.scroll_size[0]  = state.plot_size[0]
        
        state.scroll_size[1]  = ysize-state.winbuffer[1]
        state.plot_size[1] = state.scroll_size[1] > $
                             state.pixperorder*state.norders
        
;  Four cases to deal with  1) no-scroll to no-scroll.  2) scroll to no-scroll,
;  3) no-scroll to scroll, 4) scroll to scroll
        
        type = '3or4'
        
        if ~state.scrollbars and $
           state.plot_size[1] eq state.scroll_size[1] then type = '1'
        
        if state.scrollbars and $
           state.plot_size[1] eq state.scroll_size[1] then type = '2'        
        
        case type of
           
           '1': begin
              
              widget_control, state.plotwin, DRAW_XSIZE=state.plot_size[0], $
                              DRAW_YSIZE=state.plot_size[1]           
              
              widget_geom = widget_info(state.xvspec_base, /GEOMETRY)
              
              state.winbuffer[0]=widget_geom.xsize-state.plot_size[0]
              state.winbuffer[1]=widget_geom.ysize-state.plot_size[1]
              
              state.scrollbars = 0
              
           end
           
           '2': begin
              
              widget_control, state.xvspec_base,UPDATE=0
              widget_control, state.plotbase,UPDATE=0
              widget_control, state.plotwin,/DESTROY
              
              state.plotwin = widget_draw(state.plotbase,$
                                          /ALIGN_CENTER,$
                                          XSIZE=state.scroll_size[0],$
                                          YSIZE=state.scroll_size[1],$
                                          EVENT_PRO='xvspec_plotwinevent',$
                                          /KEYBOARD_EVENTS,$
                                          /BUTTON_EVENTS,$
                                          /MOTION_EVENTS,$
                                          /TRACKING_EVENTS)

              widget_control, state.plotbase,UPDATE=1              
              widget_control, state.xvspec_base,UPDATE=1
              
              widget_geom = widget_info(state.xvspec_base, /GEOMETRY)
              
              state.winbuffer[0]=widget_geom.xsize-state.scroll_size[0]
              state.winbuffer[1]=widget_geom.ysize-state.scroll_size[1]
              
              state.scrollbars = 0
              
           end
           
           '3or4': begin
              
              widget_control, state.xvspec_base,UPDATE=0
              widget_control, state.plotbase,UPDATE=0
              widget_control, state.plotwin,/DESTROY
              
              state.plotwin = widget_draw(state.plotbase,$
                                          /ALIGN_CENTER,$
                                          XSIZE=state.plot_size[0],$
                                          YSIZE=state.plot_size[1],$
                                          X_SCROLL_SIZE=state.scroll_size[0],$
                                          Y_SCROLL_SIZE=state.scroll_size[1],$
                                          /SCROLL,$
                                          EVENT_PRO='xvspec_plotwinevent',$
                                          /KEYBOARD_EVENTS,$
                                          /BUTTON_EVENTS,$
                                          /MOTION_EVENTS,$
                                          /TRACKING_EVENTS)

              widget_control, state.plotbase,UPDATE=1              
              widget_control, state.xvspec_base,UPDATE=1
              
              widget_geom = widget_info(state.xvspec_base, /GEOMETRY)
              
              state.winbuffer[0]=widget_geom.xsize-state.scroll_size[0]
              state.winbuffer[1]=widget_geom.ysize-state.scroll_size[1]
              
              state.scrollbars = 1
              
           end
           
        endcase
        
     end
     
  endcase

  wdelete,state.pixmap_wid
  window, /FREE, /PIXMAP,XSIZE=state.plot_size[0],YSIZE=state.plot_size[1]
  state.pixmap_wid = !d.window  
  
  xvspec_updatemenus,/MODE
  xvspec_plotupdate
  
end
;
;===============================================================================
;
;------------------------------Main Program------------------------------------
;
;===============================================================================
;
pro xvspec,afile,bfile,POSITION=position,LADDER=ladder,CONTINUOUS=continuous,$
           MODE=mode,PLOTWINSIZE=plotwinsize,PLOTLINMAX=plotlinmax, $
           PLOTREPLACE=plotreplace,PLOTFIX=plotfix,PLOTOPTFAIL=plotoptfail, $
           PLOTATMOSPHERE=plotatmosphere,NOUPDATE=noupdate, $
           GROUP_LEADER=group_leader,CANCEL=cancel
  
  if n_params() eq 1 then begin
     
     cancel = mc_cpar('xvspec',afile,1,'AFile',7,0)
     if cancel then return
     afile = mc_cfile(afile,CANCEL=cancel)    
     if cancel then return
     
  endif

  if n_params() eq 2 then begin
     
     cancel = mc_cpar('xvspec',bfile,1,'BFile',7,0)
     if cancel then return
     bfile = mc_cfile(bfile,CANCEL=cancel)    
     if cancel then return
     
  endif
  
  if not xregistered('xvspec') then xvspec_initcommon

  common xvspec_state
  
;  Set keyword values

  if n_elements(PLOTLINMAX) ne 0 and ~keyword_set(PLOTLINMAX) then $
     state.p.plotsatpixel = 0
  
  if n_elements(PLOTREPLACE) ne 0 and ~keyword_set(PLOTREPLACE) then $
     state.p.plotreplacepixel = 0
  
  if n_elements(PLOTFIX) ne 0 and ~keyword_set(PLOTFIX) then $
     state.p.plotfixpixel = 0
  
  if n_elements(PLOTOPTFAIL) ne 0 and ~keyword_set(PLOTOPTFAIL) then $
     state.p.plotoptfail = 0
  
  state.plotatmosphere = keyword_set(PLOTATMOSPHERE)
  
;  Now load the spectra and determine the ranges

  xvspec_loadspectra,afile,bfile,CANCEL=cancel
  if cancel then return

  if not xregistered('xvspec') then begin

;  Get plot window sizes

     if keyword_set(PLOTWINSIZE) then begin

;  Get screen size

        screensize = get_screen_size()
        
        state.scroll_size[0] = plotwinsize[0]*screensize[0]
        state.scroll_size[1] = plotwinsize[1]*screensize[1]
        state.plot_size[0] = plotwinsize[0]*screensize[0]
        state.plot_size[1] = plotwinsize[1]*screensize[1]
        
     endif

     tmpplotysize = state.scroll_size[1]>state.pixperorder*state.norders
     
     if keyword_set(MODE) then begin
        
        case mode of
           
           'Continuous': state.mode = 'Continuous'
           
           'Ladder': state.mode = 'Ladder'
           
           else:  begin
              
              print, 'Unidentified view mode requested.' + $
                     '  Defaulting to Continuous.'
              state.mode = 'Continuous'
              
           end
           
        endcase
        
     endif
       
     if keyword_set(LADDER) then state.mode = 'Ladder'
     if keyword_set(CONTINUOUS) then state.mode = 'Continuous'
     if state.norders eq 1 then state.mode = 'Continuous'

     if state.mode eq 'Ladder' then state.plot_size[1] = tmpplotysize
     state.scrollbars = state.plot_size[1] gt state.scroll_size[1]
     xvspec_mkwidget

  endif else begin

     xvspec_modwin,LADDER=ladder,CONTINUOUS=continous,PLOTWINSIZE=plotwinsize, $
                   MODE=mode,NOUPDATE=noupdate

  endelse

  *state.pspectra = *state.amspectra  
  state.pbuffer = 'A'
  state.pfile = state.afile
  
;  Get the ranges

  xvspec_getranges

;  Set the view to be flux

  state.spectype = 'Flux'
  widget_control, state.mbut_flx,/SET_BUTTON
  widget_control, state.but_flx,/SET_BUTTON
  widget_control, state.but_unc,SET_BUTTON=0
  widget_control, state.but_snr,SET_BUTTON=0
  
  state.pytitle = state.ytitle[0]
        
  *state.pyranges = *state.flxyranges
        
  state.pwrange = state.flxwrange
  state.pabswrange = state.absflxwrange
  
  state.pyrange = state.flxyrange
  state.pabsyrange = state.absflxyrange

;  Update the menus and plot the spectra
  
  xvspec_updatemenus
  xvspec_plotupdate

end
