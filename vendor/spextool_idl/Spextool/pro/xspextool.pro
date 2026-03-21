;+
; NAME:
;     xspextool
;
; PURPOSE:
;     Widget to drive the SpeX data reduction.
;    
; CATEGORY:
;     Widget
;
; CALLING SEQUENCE:
;     xspextool,FAB=fab,IPREFIX=iprefix
;    
; INPUTS:
;     None
;   
; OPTIONAL INPUTS:
;     None
; KEYWORD PARAMETERS:
;     FAB - Set to launch Spextool with the Flat and Arc Base
;     
; OUTPUTS:
;     Writes SpeX spectral FITS images to disk
;     
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCK:
;     None
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;
;-
;===============================================================================
;
pro xspextool_startup,INSTRUMENT=instrument,FAB=fab,ENG=eng,IPREFIX=iprefix,$
                      CANCEL=cancel

  cancel = 0

  common xspextool_state,state

;  Get screen size

  screensize = get_screen_size()
  
;  Get spextool and instrument information 

  mc_getspextoolinfo,spextoolpath,packagepath,spextoolkeywords,instr, $
                     notirtf,version,INSTRUMENT=instrument,CANCEL=cancel
  if cancel then return
  
;  Get paths output file name

  dir = (!version.os_family eq 'unix') ? getenv('HOME'):packagepath
  
  pathsfile = filepath('.'+strtrim(strlowcase(instr.instrument),2)+ $
                       '_path.txt',ROOT_DIR=dir)
  
;  Set the fonts 

  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return

;  Build four structures which will hold important info.
;  w - contains info pertaining to widget operations.
;  r - contains info pertaining to the reduction process.
;  d - contains all of the data.

  w = {abba2abab:0L,$
       ampcor_bg:0L,$
       arc_base:0,$             
;       atmosthresh_fld:[0L,0L],$
       autoxcor_bg:0L,$                           
       base:'',$                                  
       buttonfont:buttonfont,$
       textfont:textfont,$
       medprofile_bg:0L,$
       bdpxthresh_fld:[0,0],$                     
       calpath_fld:[0,0],$                        
;       checkseeing_bg:0L,$                       
       cals_base:0L,$                             
       combmods_bg:0L,$                           
       combflat_base:0L,$                          
       combflat_fld:[0,0],$                       
       combimgs_fld:[0,0],$                       
       combimgdir_bg:0L,$                         
       combimgmode_bg:0L,$                        
       combimgstat_dl:0L,$                                        
       combimgs_base:0,$                          
       comboname_fld:[0,0],$                      
       datapath_fld:[0,0],$                       
       eng_base:0L,$                              
       filereadmode_bg:0L,$                       
       fixbdpx_bg:0L,$                                                      
       flat_base:0,$                              
       flatfieldoffset_fld:[0L,0L],$
       flatfield_bg:0L,$                          
       flatfield_fld:[0,0],$                      
       imageinput_base:0L,$                       
       instrument:instr.instrument,$
       iprefix_fld:[0,0],$                        
       iprefix_but:0L,$                           
       iprefix_row:0L,$                           
       lc_bg:0L,$                                 
       loglin_bg:0L,$
;       maskwindow_fld:[0,0],$   
       messwin:0,$                                
       notirtf:notirtf,$                          
       optext_bg:0L,$                             
       other_base:0,$                             
       outname_fld:[0,0],$                        
       oprefix_fld:[0,0],$                       
;       oversamp_fld:[0,0],$                       
       panel:'Paths',$
       paltspecbias_bg:0L,$
       path_base:0,$                              
       path_but:0,$
       plotprofiles:0L,$                          
       plotxcor_bg:0L,$                           
       procpath_fld:[0,0],$                       
       psapfindcur_base:0,$                       
       psapfindtrace_base:0,$                     
       psappos_fld:[0,0],$                        
       psapradius_fld:[0,0],$                     
       psapsigns_fld:[0L,0L],$                    
       psbginfo_base:0,$                          
       psbgstart_fld:[0,0],$                      
       psbgsub_bg:0L,$                            
       psbgwidth_fld:[0,0],$                      
       psbgfitdeg_fld:[0,0],$                    
       pscol2_base:0,$                            
       psfradius_fld:[0,0],$                      
       psitracefile_fld:[0,0],$                   
       psnaps_bg:0,$                              
       psorder_bg:0L,$                            
       psotracefile_fld:[0,0],$                   
       pstrace_base:0,$                           
       ps_base:0,$                                
       rectmethod_bg:0L,$
       reductionmodes:['A','A-B','A-Sky/Dark'],$       
       reducmode_bg:0L,$        
       combimgthresh_fld:[0L,0L],$                
       sky_base:0,$                               
       skycombthresh_fld:[0L,0L],$                
       skyflat_fld:[0,0],$                        
       skyimage_base:0,$                          
       skyimage_fld:[0,0],$                       
       skyoname_fld:[0,0],$                       
       skyimages_fld:[0,0],$ 
       skystat_dl:0L,$                            
       slitw_arc_fld:[0,0],$                      
       slitw_pix_fld:[0,0],$                      
       sourceimages_fld:[0,0],$                   
;       storedwc_bg:0L,$                           
       sumap_fld:[0,0],$                          
       table:0L,$                                 
       tracedeg_fld:[0,0],$                       
       tracestepsize_fld:[0,0],$                  
       tracesigthresh_fld:[0L,0L],$                 
       tracewinthresh_fld:[0L,0L],$   
       useappos:0L,$
       version:version[0],$                       
       wavecalthresh_fld:[0,0],$                  
       wavecal_fld:[0,0],$                        
       wconame_fld:[0,0],$                        
       xsapfindman_base:0,$                       
       xsapfindtrace_base:0,$                     
       xsappos_fld:[0,0],$                        
       xsapradius_fld:[0,0],$                     
       xsbgfitdeg_fld:[0,0],$                    
       xsbginfo_base:0,$                          
       xsbgsub_bg:0L,$                            
       xsbg_fld:[0,0],$                           
       xs_base:0,$                                
       xsapfindcur_base:0L,$
       xsapfindkey_base:0L,$
       xsitracefile_fld:[0,0],$                   
       xstrace_base:0,$                           
       xsnaps_fld:[0,0],$
       xspextool_base:0L}

  r = {abba2abab:0,$
;       airmass:instr.airmass,$
       ampcor:instr.ampcorrect,$
       aph_arc:1.0,$
       appos:ptr_new(fltarr(2)),$
       axranges:ptr_new(2),$
       tmpappos:ptr_new(fltarr(2)),$
       apsign:ptr_new(intarr(2)),$
       arccombinestat:'Median',$
       arcreductionmode:'ShortXD',$
;       autoxcorr:instr.autoxcorr,$
       bdpxmask:instr.bdpxmk,$
       medprofile:instr.aveprof,$
       calpath:'',$
;       checkseeing:instr.checkseeing,$
       combsclordrs:0L,$
       combbgsub:0L,$
       combimgstat:0,$
       combimgdir:instr.combimgdir,$
       combreductionmode:instr.combimgmode,$
;       combskystat:0,$
       combstats:['Robust Weighted Mean', $       
                  'Robust Mean (RMS)', $
                  'Robust Mean (Std Error)', $
                  'Weighted Mean',$
                  'Mean (RMS)', $
                  'Mean (Std Error)', $
                  'Median (MAD)', $
                  'Median (Std Error)' + $
                  '','Sum'],$
       datapath:'',$
       detsol:0,$
       doallsteps:0,$
       edgecoeffs:ptr_new(fltarr(2,2)),$
       fitsreadprog:instr.fitsreadprogram,$
       fixbdpx:instr.fixbdpx,$
       filereadmode:instr.filereadmode,$
       flatfield:instr.flatfield,$
       guessappos:ptr_new(2),$
       hdrinfo:ptr_new(intarr(2)),$
;       ha:instr.ha,$
       instr:instr.instrument,$
;       irafname:instr.irafname,$
;       itot:ptr_new(2),$
;       juldate:0.0,$
       keywords:instr.xspextool_keywords,$
       lc:instr.lincorrect,$
       mk1dlinefindplot:0,$
       mk2dlinefitsplot:0,$
       mkloglinplot:0L,$
       mklinesummaryplot:0,$
       modename:'',$,$
       ncols:instr.ncols,$
       nint:instr.nint,$
       nsuffix:instr.nsuffix,$
       norders:0L,$
;       normflat:normflat,$
       ntotaps:4,$
       nrows:instr.nrows,$
       obsmode:'',$
       offset:0.,$
       optext:instr.optextract,$
       orders:ptr_new([0,1,2,3,4,5]),$
       paltspecbias:0L,$
       packagepath:packagepath,$
       pathsfile:pathsfile,$
       plotautoxcorr:instr.plotxcorr,$
       plotresiduals:0,$
       plotwinsize:long(instr.plotwinsize*screensize),$
;       posangle:instr.posangle,$
       prefix:'',$
       procpath:'',$
;       profcoeffs:ptr_new(fltarr(2)),$
       ps:0.0,$
;       aveprofcoeffs:ptr_new(fltarr(2)),$
       psbgsub:instr.psbgsub,$
       psapfindmode:'Cursor',$
       psapfindmanmode:'Auto',$
       pscontinue:0L,$
       psdoorders:ptr_new([1]),$
       psnaps:instr.psnaps,$
       rectmethodmode:instr.rectmethod,$
       reductionmode:instr.reductionmode,$
       rp:0L,$                                      
       rms:ptr_new(fltarr(2)),$
       rotation:0,$
       lincormax:instr.lincormax,$
       slith_arc:0.,$
       slith_pix:0.,$
       slitw_arc:0.,$
       slitw_pix:0.,$
       sourcedir:'raw/',$
       sourcepath:'',$
       spatfmt:'',$
       spextoolpath:spextoolpath,$
       stdimage:instr.stdimage,$
       subscatlgt:0,$
       suffix:instr.suffix,$
       tablesize:10,$                              
;       time:instr.time,$
       tracecoeffs:ptr_new(fltarr(2)),$
       usestoredwc:0L,$
       useappositions:0L,$
       wavecal:0,$
       calmodule:instr.calmodule,$
       wavefmt:'',$
       workfiles:ptr_new(strarr(2)),$
       wctype:'',$
       xranges:ptr_new(fltarr(2)),$
       xsapfindmode:'Cursor',$
       xsbgsub:instr.xsbgsub,$
       xscontinue:0L,$
       xsdoorders:ptr_new([1]),$
       xsnaps:0L,$
       profiles:ptr_new(fltarr(2))}

  d = {awave:ptr_new(2),$
       bdpxmask:ptr_new(2),$
       flat:ptr_new(2),$
;       xo2w:ptr_new(2),$
;       xo2wdeg:[1,1],$
       homeorder:0,$
       atrans:ptr_new(2),$
;       xy2wdeg:[1,1],$
;       xy2sdeg:[1,1],$
       disp:ptr_new(fltarr(2)),$
       mask:ptr_new(fltarr(2)),$
       omask:ptr_new(fltarr(2)),$
       bitmask:ptr_new(fltarr(2)),$
       spectra:ptr_new(fltarr(2)),$
       varimage:ptr_new(fltarr(2)),$
       wavecal:ptr_new(fltarr(2)),$
       spatcal:ptr_new(fltarr(2)),$
       indices:ptr_new(2),$
       imgstruc:ptr_new(2),$
       varstruc:ptr_new(2),$
       bpmstruc:ptr_new(2),$
;       indices:ptr_new(2),$
;       xy2w:ptr_new(fltarr(2)),$
;       xy2s:ptr_new(fltarr(2)),$
;       tri:ptr_new(2),$
       workimage:ptr_new(fltarr(2,2))}

;  Build the widget.

  w.xspextool_base = widget_base(/COLUMN,$
                                 TITLE='Spextool '+w.version+' for '+ $
                                 instr.instrument)
  
     button = widget_button(w.xspextool_base,$
                            FONT=buttonfont,$
                            VALUE='Quit',$
                            UVALUE='Quit')

   mess = 'Welcome to Spextool '+w.version+'.'
   w.messwin = widget_text(w.xspextool_base, $
                           FONT=textfont,$
                           VALUE=mess, $
                           YSIZE=2)

; Main base, which is always showing.

     main_base = widget_base(w.xspextool_base,$
                             /ROW,$
                             FRAME=5)
  
        col1_base = widget_base(main_base,$
                                /COLUMN,$
;                                /BASE_ALIGN_RIGHT,$
                                FRAME=2)
        
           w.filereadmode_bg = cw_bgroup(col1_base,$
                                         ['Filename','Index'],$
                                         /ROW,$
                                         LABEL_LEFT='File Read Mode:',$
                                         /RETURN_NAME,$
                                         /NO_RELEASE,$
                                         UVALUE='File Readmode',$
                                         FONT=buttonfont,$
                                         /EXCLUSIVE)
           widget_control,w.filereadmode_bg, $
                          SET_VALUE=(r.filereadmode eq 'Index') ? 1:0

           row = widget_base(col1_base,$
                             /ALIGN_RIGHT,$
                             /ROW,$
                             /BASE_ALIGN_RIGHT)
           
              w.iprefix_row = widget_base(row)
           
              desc = [ '1\Input Prefix']
           
              w.iprefix_but = cw_pdmenu(w.iprefix_row, desc, $
                                        UVALUE='Input Prefix Button',$
                                        FONT=buttonfont,$
                                        /RETURN_NAME)
           
              fld = coyote_field2(row,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE=':',$
                                  UVALUE='Input Prefix',$
                                  XSIZE=27,$
                                  TEXTID=textid)
              w.iprefix_fld = [fld,textid]

           row = widget_base(col1_base,$
                             /ROW,$
                             /ALIGN_RIGHT,$
                             /BASE_ALIGN_RIGHT)

              fld = coyote_field2(row,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE='Output Prefix:',$
                                  UVALUE = 'Output Prefix',$
                                  XSIZE=27,$
                                  VALUE=instr.oprefix,$
                                  TEXTID=textid)
              w.oprefix_fld = [fld,textid]

           row = widget_base(col1_base,$
                             /ROW,$
                             /ALIGN_RIGHT,$
                             /BASE_ALIGN_RIGHT)
              
              fld = coyote_field2(row,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE='Output File Name(s):',$
                                  UVALUE='Output File Name',$
                                  XSIZE=27,$
                                  TEXTID=textid)
              w.outname_fld = [fld,textid]
              
           w.plotprofiles = widget_label(col1_base,$
                                         UVALUE='Plot Profiles',$
                                         VALUE='')
           
        w.imageinput_base = widget_base(main_base,$
                                        /COLUMN,$
                                        FRAME=2,$
                                        MAP=0)
     
           w.reducmode_bg = cw_bgroup(w.imageinput_base,$
                                      FONT=buttonfont,$
                                      w.reductionmodes,$
                                      /ROW,$
                                      /RETURN_NAME,$
                                      /NO_RELEASE,$
                                      /EXCLUSIVE,$
                                      LABEL_LEFT='Reduction Mode:',$
                                      UVALUE='Reduction Mode')

           widget_control, w.reducmode_bg, $
                SET_VALUE=where(w.reductionmodes eq instr.reductionmode)

           base = widget_base(w.imageinput_base,$
                              /COLUMN,$
                              /BASE_ALIGN_RIGHT)

              row = widget_base(base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)


                 cbox = widget_combobox(row,$
                                        FONT=buttonfont,$
                                        VALUE=['raw/','cal/','proc/'],$
                                        SCR_XSIZE=70,$
                                        UVALUE='Source Images Directory')
           
                 button = widget_button(row,$
                                        FONT=buttonfont,$
                                        VALUE='Images',$                
                                        UVALUE='Source Images Button')
                                  
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE=':',$
;                                     VALUE='23',$
                                     UVALUE='Source Images',$
                                     XSIZE=19,$
                                     TEXTID=textid)
                 w.sourceimages_fld = [fld,textid]
                 
              w.skyimage_base = widget_base(base,$
                                            /ROW,$
                                            /BASE_ALIGN_CENTER)

                 button = widget_button(w.skyimage_base,$
                                        FONT=buttonfont,$
                                        VALUE='Full Sky/Dark Image',$
                                        UVALUE='Sky Image Button')
                 
                 fld = coyote_field2(w.skyimage_base,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE=':',$
                                     UVALUE='Super Sky Image',$
;                                     VALUE='dark121-125.fits',$
                                     XSIZE=19,$
                                     TEXTID=textid)
                 w.skyimage_fld = [fld,textid]

              row = widget_base(base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 button = widget_button(row,$
                                        FONT=buttonfont,$
                                        VALUE='Full Flat Name',$
                                        UVALUE='Full Flat Name Button')
                 
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE=':',$
;                                     VALUE='flat48-52.fits',$
                                     UVALUE='Full Flat Name',$
                                     XSIZE=19,$
                                     TEXTID=textID)
                 w.flatfield_fld = [fld,textID]
                 
              row = widget_base(base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 button = widget_button(row,$
                                        FONT=buttonfont,$
                                        VALUE='Full Wavecal Name',$           
                                        UVALUE='Full Wavecal Name Button')
                 
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE=':',$
;                                     VALUE='wavecal53-54.fits',$
                                     UVALUE='Full Wavecal Name',$
                                     XSIZE=19,$
                                     TEXTID=textid)
                 w.wavecal_fld = [fld,textid]

                 row = widget_base(w.imageinput_base,$
                                  /ROW)

                    button = widget_button(row,$
                                           FONT=buttonfont,$
                                           VALUE='Load Image',$
                                           UVALUE='Load Single Image')

                    button = widget_button(row,$
                                           FONT=buttonfont,$
                                           /ALIGN_RIGHT,$
                                           VALUE='Do All Steps',$
                                           UVALUE='Do All Steps')

;                    button = widget_button(row,$
 ;                                          FONT=buttonfont,$
 ;                                          /ALIGN_RIGHT,$
;                                           VALUE='Save Orders',$
;                                           UVALUE='Save Orders')
                    
; Menu bar

     menubar = widget_base(w.xspextool_base,$
                           EVENT_PRO='xspextool_menuevent',$
                           /ROW)      

     TITLES=['Paths','Cals','Combine Images','Point Source', $
             'Extended Source','Other','Eng.']

     sub = (keyword_set(ENG) eq 1) ? 1:2
     
        row = widget_base(menubar,$
                          /ROW,$
                          /TOOLBAR,$
                          /EXCLUSIVE,$
                          /BASE_ALIGN_CENTER)

           for i = 0,n_elements(TITLES)-sub do begin

              button = widget_button(row,$
                                     VALUE=TITLES[i],$
                                     UVALUE=TITLES[i],$
                                     /NO_RELEASE,$
                                     FONT=buttonfont)
              
              if i eq 0 then begin

                 w.path_but = button
                 widget_control, button, /SET_BUTTON

              endif

           endfor

            button = widget_button(menubar,$
                                   VALUE='Help',$
                                   UVALUE='Help',$
                                   FONT=buttonfont)

;  Now build the work spaces.

   work_base = widget_base(w.xspextool_base,$
                           FRAME=5)

; Paths base.

   fieldwidth = 70
      w.path_base = widget_base(work_base,$
                                /COLUMN,$
                                /MAP)
      
         row = widget_base(w.path_base,$
                           /ROW,$
                           /BASE_ALIGN_CENTER)
         
            button = widget_button(row,$
                                   FONT=buttonfont,$
                                   XSIZE=110,$                             
                                   VALUE='Raw Path',$
                                   UVALUE='Data Path Button')
            
            fld = coyote_field2(row,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE=':',$
                                UVALUE='Data Path Field',$
                                XSIZE=fieldwidth,$
                                TEXTID=textid)
            w.datapath_fld = [fld,textid]
            
            button = widget_button(row,$
                                   FONT=buttonfont,$
                                   VALUE='Clear',$
                                   UVALUE='Clear Data Path')
            
            row = widget_base(w.path_base,$
                              /ROW,$
                              /BASE_ALIGN_CENTER)
            
            button = widget_button(row,$
                                   FONT=buttonfont,$
                                   XSIZE=110,$
                                   VALUE='Cal Path',$
                                   UVALUE='Cal Path Button')
            
            fld = coyote_field2(row,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE=':',$
                                UVALUE='Cal Path Field',$
                                XSIZE=fieldwidth,$
                                TEXTID=textid)
            w.calpath_fld = [fld,textid]
            
            button = widget_button(row,$
                                   FONT=buttonfont,$
                                   VALUE='Clear',$
                                   UVALUE='Clear Cal Path')
            
         row = widget_base(w.path_base,$
                           /ROW,$
                           /BASE_ALIGN_CENTER)
         
            button = widget_button(row,$
                                   FONT=buttonfont,$
                                   XSIZE=110,$
                                   VALUE='Proc Path',$
                                   UVALUE='Proc Path Button')
            
            fld = coyote_field2(row,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE=':',$
                                UVALUE='Proc Path Field',$
                                XSIZE=fieldwidth,$
                                TEXTID=textid)
            w.procpath_fld = [fld,textid]
            
            clear = widget_button(row,$
                                  FONT=buttonfont,$
                                  VALUE='Clear',$
                                  UVALUE='Clear Proc Path')
                     
;  Cal Base

      w.cals_base = widget_base(work_base,$
                                /COLUMN,$
                                MAP=0)

      call_procedure,instr.calmodule,w.cals_base,FAB=fab,ENG=eng,CANCEL=cancel
      if cancel then return
      
;  Combine Images base

      w.combimgs_base = widget_base(work_base,$
                                    /COLUMN,$
                                    MAP=0)
      
         row_base = widget_base(w.combimgs_base,$
                                /ROW)
         
            col1_base = widget_base(row_base,$
                                    FRAME=2,$
                                    /COLUMN)
            
               row = widget_base(col1_base,$
                                 /ROW,$
                                 /BASE_ALIGN_CENTER)

                  w.combimgmode_bg = cw_bgroup(row,$
                                               FONT=buttonfont,$
                                            ['A','A-B'],$
                                               /ROW,$
                                               /NO_RELEASE,$
                                               /EXCLUSIVE,$
                                               /RETURN_NAME,$
                                           LABEL_LEFT='Beam Switching Mode:',$
                                               UVALUE='Combine Reduction Mode')

                  widget_control,w.combimgmode_bg, $
                              SET_VALUE=where(['A','A-B'] eq instr.combimgmode) 

                  if w.notirtf then begin
                  
                     w.abba2abab = cw_bgroup(row,$
                                             ['ABBA to ABAB'],$
                                             FONT=buttonfont,$
                                             UVALUE='ABBA to ABAB',$
                                             /NONEXCLUSIVE)
                     widget_control,w.abba2abab, MAP=0

                  endif

               row = widget_base(col1_base,$
                                 /ROW,$
                                 /BASE_ALIGN_CENTER)
               
                  button = widget_button(row,$
                                         FONT=buttonfont,$
                                         VALUE='Images',$
                                         UVALUE='Combine Images Button')

                  fld = coyote_field2(row,$
                                      LABELFONT=buttonfont,$
                                      FIELDFONT=textfont,$
                                      TITLE=':',$
;                                      VALUE='39-44',$
                                      UVALUE='Combine Images Field',$
                                      XSIZE=70,$
                                      TEXTID=textid) 
                  w.combimgs_fld = [fld,textid]
               
               row = widget_base(col1_base,$
                                 /ROW,$
                                 /BASE_ALIGN_CENTER)

                  w.combmods_bg = cw_bgroup(row,$
                                             FONT=buttonfont,$
                                             ['Scale Orders','Column BG Sub'],$
                                             /ROW,$
                                             /NONEXCLUSIVE,$
                                             LABEL_LEFT='Modifications:',$
                                       SET_VALUE=[r.combsclordrs,r.combbgsub],$
                                             UVALUE='Combine Images Mods')

                  w.combflat_base = widget_base(row,$
                                               /ROW,$
                                               /BASE_ALIGN_CENTER)
                  
                     button = widget_button(w.combflat_base,$
                                            FONT=buttonfont,$
                                            VALUE='Full Flat Name',$
                                            UVALUE='Combine Flat Field Button')
                     
                     fld = coyote_field2(w.combflat_base,$
                                         LABELFONT=buttonfont,$
                                         FIELDFONT=textfont,$
                                         TITLE=':',$
                                         UVALUE='Combine Flat Field',$
                                         XSIZE=15,$
                                         TEXTID=textid) 
                     w.combflat_fld = [fld,textid]
                     
                  widget_control, w.combflat_base,SENSITIVE=0


               row = widget_base(col1_base,$
                                 /ROW,$
                                 /BASE_ALIGN_CENTER)
               
                  w.combimgstat_dl = widget_droplist(row,$
                                                     FONT=buttonfont,$
                                                     TITLE='Statistic:',$
                                                     VALUE=r.combstats,$
                                        UVALUE='Image Combination Statistic')
                  r.combimgstat = where(r.combstats eq instr.combimgstat)
                  widget_control, w.combimgstat_dl, $
                                  SET_DROPLIST_SELECT=r.combimgstat
                    
                  fld = coyote_field2(row,$
                                      LABELFONT=buttonfont,$
                                      FIELDFONT=textfont,$
                                      TITLE='Thresh:',$
                                      UVALUE='Robust Threshold',$
                                      XSIZE=5,$
                                      VALUE=instr.combimgthresh,$
                                      TEXTID=textid)
                  w.combimgthresh_fld = [fld,textid]
                  
               row = widget_base(col1_base,$
                                 /ROW,$
                                 /BASE_ALIGN_CENTER)
               
                  w.combimgdir_bg = cw_bgroup(row,$
                                              FONT=buttonfont,$
                                              ['cal/','proc/'],$
                                              /ROW,$
                                              /RETURN_NAME,$
                                              /NO_RELEASE,$
                                              /EXCLUSIVE,$
                                              LABEL_LEFT='Output Name:',$
                                              UVALUE='Image Combine Directory')
                  widget_control, w.combimgdir_bg, $
                         SET_VALUE=where(['cal/','proc/'] eq instr.combimgdir)


                  fld = coyote_field2(row,$
                                      LABELFONT=buttonfont,$
                                      FIELDFONT=textfont,$
                                      TITLE='',$
                                      UVALUE='Combine Output Name',$
                                      XSIZE=15,$
;                                      VALUE='junk',$
                                      textid=textid) 
                  w.comboname_fld = [fld,textid]
                  
               button = widget_button(col1_base,$
                                      FONT=buttonfont,$
                                      VALUE='Combine Images',$
                                      UVALUE='Combine Images')
         
;  Point Source Base

      w.ps_base = widget_base(work_base,$
                              /COLUMN,$
                              MAP=0)
      
         row1_base = widget_base(w.ps_base,$
                                 /ROW)
         
            col1_base = widget_base(row1_base,$
                                    /COLUMN,$
                                    FRAME=2)
            
               label = widget_label(col1_base,$
                                    VALUE='1. Identify Apertures',$
                                    FONT=buttonfont,$
                                    /ALIGN_LEFT)
               
               button = widget_button(col1_base,$
                                      FONT=buttonfont,$
                                      VALUE='Make Spatial Profiles',$
                                      UVALUE='Make Spatial Profiles')
               
;               bg = cw_bgroup(col1_base,$
;                              FONT=buttonfont,$
;                              ['Cursor','Import Trace'],$
;                              /EXCLUSIVE,$
;                              /ROW,$
;                              /RETURN_NAME,$
;                              /NO_RELEASE,$
;                              LABEL_LEFT='Method:',$
;                              SET_VALUE=0,$
;                              UVALUE='PS Object Find Mode')
               
               base1 = widget_base(col1_base)

                  w.psapfindcur_base = widget_base(base1,$
                                                   /COLUMN,$
                                                   /BASE_ALIGN_LEFT,$
                                                   MAP=1)   
                  
                     bf = cw_bgroup(w.psapfindcur_base,$
                                    FONT=buttonfont,$
                                    ['Auto','Guess','Fix'],$
                                    /EXCLUSIVE,$
                                    LABEL_LEFT='Find Type:',$
                                    /ROW,$
                                    /RETURN_NAME,$
                                    /NO_RELEASE,$
                                    SET_VALUE=0,$
                                    UVALUE='PS Manual Find Mode')

                     w.psnaps_bg = cw_bgroup(w.psapfindcur_base,$
                                             FONT=buttonfont,$
                                             ['1','2','3','4'],$
                                             LABEL_LEFT='N Apertures:',$
                                             /EXCLUSIVE,$
;                                             /NONEXCLUSIVE,$
                                             /ROW,$
                                             /RETURN_INDEX,$
                                             /NO_RELEASE,$
                                             SET_VALUE=r.psnaps-1,$
                                             UVALUE='PS N Apertures')

               button = widget_button(col1_base,$
                                      FONT=buttonfont,$
                                      VALUE='Find Aperture Positions',$
                                      UVALUE='Find Aperture Positions')
                     
                  w.psapfindtrace_base = widget_base(base1,$
                                                     /COLUMN,$
                                                     MAP=0) 

                     row = widget_base(w.psapfindtrace_base,$
                                       /ROW,$
                                      /BASE_ALIGN_CENTER)

                        button = widget_button(row,$
                                               FONT=buttonfont,$
                                               VALUE='Full Trace File',$
                                               UVALUE='Trace File Button')
                     
                        fld = coyote_field2(row,$
                                            LABELFONT=buttonfont,$
                                            FIELDFONT=textfont,$
                                            TITLE=':',$
                                            UVALUE='PS Trace File',$
                                            XSIZE=15,$
;                                         VALUE='trace.dat',$
                                            TEXTID=textid) 
                        w.psitracefile_fld = [fld,textid]
                        
                     button = widget_button(w.psapfindtrace_base,$
                                            FONT=buttonfont,$
                                            VALUE='Load Trace File',$
                                            UVALUE='Load Trace File')
                     
            w.pstrace_base = widget_base(row1_base,$
                                         /COLUMN,$              
                                         FRAME=2)
            
               label = widget_label(w.pstrace_base,$
                                    FONT=buttonfont,$
                                    VALUE='2. Trace Objects',$
                                    /ALIGN_LEFT)
                              
              w.useappos = cw_bgroup(w.pstrace_base,$
                                     FONT=buttonfont,$
                                     'Use Ap Positions',$
                                     /NONEXCLUSIVE,$
                                     /COLUMN,$
                                     SET_VALUE=[0],$
                                     UVALUE='Use Ap Positions')

;               fld = coyote_field2(w.pstrace_base,$
;                                   LABELFONT=buttonfont,$
;                                   FIELDFONT=textfont,$
;                                   TITLE='Filename:',$
;                                   UVALUE='PS Trace Filename',$
;                                   XSIZE=10,$
;                                   TEXTID=textid)
;               w.psotracefile_fld = [fld,textid]

               button = widget_button(w.pstrace_base,$
                                      FONT=buttonfont,$
                                      VALUE='Trace Objects',$
                                      UVALUE='PS Trace Objects')
               
           pscol5_base = widget_base(row1_base,$
                                      /COLUMN,$                                
                                      FRAME=2)

               label = widget_label(pscol5_base,$
                                    FONT=buttonfont,$
                                    VALUE='3. Define Extraction Parameters',$
                                    /ALIGN_LEFT)     

               row = widget_base(pscol5_base,$
                                 /ROW)
               
               w.optext_bg = cw_bgroup(row,$
                                       FONT=buttonfont,$
                                       ['Optimal'],$
                                       /ROW,$
                                       /NONEXCLUSIVE,$
                                       SET_VALUE=[1],$
                                       UVALUE='Optimal Extraction')
                           
               w.medprofile_bg = cw_bgroup(row,$
                                           FONT=buttonfont,$
                                           'Average Profile',$
                                           /ROW,$
                                           /NONEXCLUSIVE,$
                                           SET_VALUE=r.medprofile[0],$
                                           UVALUE='Median Profile')



               row = widget_base(pscol5_base,$
                                 /ROW)

                  fld = coyote_field2(row,$
                                      LABELFONT=buttonfont,$
                                      FIELDFONT=textfont,$
                                      TITLE='PSF / Ap Radius:',$
                                      UVALUE='PSF Aperture',$
                                      VALUE=instr.pspsfrad,$
                                      XSIZE=4,$
                                      TEXTID=textid)
                  w.psfradius_fld = [fld,textid]                       
               
                  fld = coyote_field2(row,$
                                      LABELFONT=buttonfont,$
                                      FIELDFONT=textfont,$
                                      TITLE='/ ',$
                                      UVALUE='PS Aperture',$
                                      VALUE=instr.psaprad,$
                                      XSIZE=4,$
                                      textid=textid)
                  w.psapradius_fld = [fld,textid]

               w.psbgsub_bg = cw_bgroup(pscol5_base,$
                                        FONT=buttonfont,$
                                        'BG Sub On',$
                                        /NONEXCLUSIVE,$
                                        /COLUMN,$
                                        SET_VALUE=[r.psbgsub],$
                                        UVALUE='PS BG Subtraction')

               w.psbginfo_base = widget_base(pscol5_base,$
                                             /COLUMN)
               
                  row = widget_base(w.psbginfo_base,$
                                    /ROW)

                     fld = coyote_field2(row,$
                                         LABELFONT=buttonfont,$
                                         FIELDFONT=textfont,$
                                         TITLE='BG Start / Width:',$
                                         UVALUE='PS Bg Start',$
                                         VALUE=instr.psbgstart,$,$
                                         XSIZE=4,$
                                         TEXTID=textid)
                     w.psbgstart_fld = [fld,textid]
                  
                     fld= coyote_field2(row,$
                                        LABELFONT=buttonfont,$
                                        FIELDFONT=textfont,$
                                        TITLE='/ ',$
                                        UVALUE='PS Bg Width',$
                                        XSIZE=3,$
                                        VALUE=instr.psbgwidth,$
                                        TEXTID=textid)                
                     w.psbgwidth_fld  = [fld,textid]

                  fld = coyote_field2(w.psbginfo_base,$
                                      LABELFONT=buttonfont,$
                                      FIELDFONT=textfont,$
                                      TITLE='Fit Degree:',$
                                      UVALUE='PS Bg Fit Order',$
                                      XSIZE=3,$
                                      VALUE=instr.psbgdeg,$
                                      TEXTID=textID)
                  w.psbgfitdeg_fld = [fld,textID]
                  
               
               button = widget_button(pscol5_base,$
                                      FONT=buttonfont,$
                                      VALUE='Show Apertures',$
                                      UVALUE='PS Show Apertures')


               button = widget_button(w.ps_base,$
                                     FONT=buttonfont,$
                                      VALUE='Extract Spectra',$
                                      UVALUE='PS Extract Spectra')

;  Extended Source Base

      w.xs_base = widget_base(work_base,$
                              /COLUMN,$
                              MAP=0)

         row1_base = widget_base(w.xs_base,$
                                 /ROW)

            col1_base = widget_base(row1_base,$
                                    /COLUMN,$
                                    FRAME=2)
            
               label = widget_label(col1_base,$
                                    VALUE='1. Identify Apertures',$
                                    FONT=buttonfont,$
                                    /ALIGN_LEFT)
               
               button = widget_button(col1_base,$
                                      FONT=buttonfont,$
                                      VALUE='Make Spatial Profiles',$
                                      UVALUE='Make Spatial Profiles')

               bg = cw_bgroup(col1_base,$
                              FONT=buttonfont,$
                              ['Cursor','Keyboard'],$
                              /EXCLUSIVE,$
                              /ROW,$
                              /RETURN_NAME,$
                              /NO_RELEASE,$
                              LABEL_LEFT='Method:',$
                              SET_VALUE=0,$
                              UVALUE='XS Object Find Mode')               

               base1 = widget_base(col1_base)

                  w.xsapfindcur_base = widget_base(base1,$
                                                   /COLUMN,$
                                                   /BASE_ALIGN_LEFT,$
                                                   MAP=1)

                  fld = coyote_field2(w.xsapfindcur_base,$
                                      LABELFONT=buttonfont,$
                                      FIELDFONT=textfont,$
                                      TITLE='Number of Apertures:',$
                                      UVALUE='XS N Apertures',$
                                      VALUE='',$
                                      XSIZE=10,$
                                      TEXTID=textid)
                  w.xsnaps_fld = [fld,textid]  

                  button = widget_button(w.xsapfindcur_base,$
                                         FONT=buttonfont,$
                                         VALUE='Identify Aperture Positions',$ 
                                       UVALUE='XS Identify Aperture Positions')
                                    
                  w.xsapfindkey_base = widget_base(base1,$
                                                   /COLUMN,$
                                                   /BASE_ALIGN_LEFT,$
                                                   MAP=0)

                     fld = coyote_field2(w.xsapfindkey_base,$
                                         LABELFONT=buttonfont,$
                                         FIELDFONT=textfont,$
                                         TITLE='Positions:',$
                                         UVALUE='XS Aperture Position',$
                                         VALUE='',$
                                         XSIZE=25,$
                                         TEXTID=textid)
                     w.xsappos_fld = [fld,textid]  

                     button = widget_button(w.xsapfindkey_base,$
                                            FONT=buttonfont,$
                                            VALUE='Store Aperture Positions',$ 
                                       UVALUE='XS Identify Aperture Positions')

            w.xstrace_base = widget_base(row1_base,$
                                         /COLUMN,$              
                                         FRAME=2)
            
               label = widget_label(w.xstrace_base,$
                                    FONT=buttonfont,$
                                    VALUE='2. Trace Objects',$
                                    /ALIGN_LEFT)
                              
               button = widget_button(w.xstrace_base,$
                                      FONT=buttonfont,$
                                      VALUE='Create Trace',$
                                      UVALUE='XS Trace Objects')
               
            xscol4_base = widget_base(row1_base,$
                                      /COLUMN,$                                
                                      FRAME=2)
            
               label = widget_label(xscol4_base,$
                                    FONT=buttonfont,$
                                    VALUE='3. Define Extraction Parameters',$
                                    /ALIGN_LEFT)  
                                           
               fld = coyote_field2(xscol4_base,$
                                   LABELFONT=buttonfont,$
                                   FIELDFONT=textfont,$
                                   TITLE='Ap Radius:',$
                                   UVALUE='XS Aperture',$
                                   VALUE='1,1',$
                                   XSIZE=20,$
                                   TEXTID=textid)
               w.xsapradius_fld = [fld,textid]

              w.xsbgsub_bg = cw_bgroup(xscol4_base,$
                                        FONT=buttonfont,$
                                        'BG Sub On',$
                                        /NONEXCLUSIVE,$
                                        /COLUMN,$
                                        SET_VALUE=[r.xsbgsub],$
                                        UVALUE='XS BG Subtraction')
              

              fld = coyote_field2(xscol4_base,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE='Bg Reg:',$
                                  UVALUE='XS Bg Regions',$
                                  XSIZE=30,$
                                  VALUE=instr.xsbgreg,$
                                  TEXTID=textid)
              w.xsbg_fld = [fld,textid]

              fld = coyote_field2(xscol4_base,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE='Bg Fit Degree:',$
                                  UVALUE='XS Bg Fit Order',$
                                  XSIZE=8,$
                                  VALUE=instr.xsbgdeg,$
                                  TEXTID=textID)
              w.xsbgfitdeg_fld = [fld,textID]
              


               
               button = widget_button(xscol4_base,$
                                      FONT=buttonfont,$
                                      VALUE='Show Apertures',$
                                      UVALUE='XS Define Apertures')
               
           button = widget_button(w.xs_base,$
                                  FONT=buttonfont,$
                                  VALUE='Extract Spectra',$
                                  UVALUE='XS Extract Spectra')

;  Other Base

   w.other_base = widget_base(work_base,$
                              /ROW,$
                              MAP=0)

      row_base = widget_base(w.other_base,$
                             /ROW)

         base = widget_base(row_base,$
                            /COLUMN,$
                            /BASE_ALIGN_LEFT,$
                            FRAME=2)
         
            label = widget_label(base,$
                                 /ALIGN_LEFT,$
                                 VALUE='Reduction Steps:',$
                                 FONT=buttonfont)

            w.plotxcor_bg = cw_bgroup(base,$
                                      FONT=buttonfont,$
                                      ['Plot X-Correlate'],$
                                      /ROW,$
                                      /RETURN_NAME,$
                                      /NONEXCLUSIVE,$
                                      UVALUE='Plot Auto X-Correlate',$
                                      SET_VALUE=r.plotautoxcorr[0])
            widget_control, w.plotxcor_bg,SENSITIVE=r.plotautoxcorr[1]
                        
            w.ampcor_bg = cw_bgroup(base,$
                                    FONT=buttonfont,$
                                    ['Amp Correction'],$
                                    /ROW,$
                                    /NONEXCLUSIVE,$
                                    SET_VALUE=r.ampcor[0],$
                                    UVALUE='Amp Correction')
            widget_control, w.ampcor_bg,SENSITIVE=r.ampcor[1]

            
            w.lc_bg = cw_bgroup(base,$
                                FONT=buttonfont,$
                                ['Linearity Correction'],$
                                /ROW,$
                                /NONEXCLUSIVE,$
                                SET_VALUE=r.lc[0],$
                                UVALUE='Linearity Correction')
            widget_control, w.lc_bg,SENSITIVE=r.lc[1]


            w.flatfield_bg = cw_bgroup(base,$
                                       FONT=buttonfont,$
                                       ['Flat Field'],$
                                       /ROW,$
                                       /NONEXCLUSIVE,$
                                       SET_VALUE=r.flatfield[0],$
                                       UVALUE='Flat Field')
            widget_control, w.flatfield_bg,SENSITIVE=r.flatfield[1]

            w.rectmethod_bg = cw_bgroup(base,$
                                        FONT=buttonfont,$
                                        ['Interp','Polyclip'],$
                                        /NO_RELEASE,$
                                        /ROW,$
                                        /EXCLUSIVE,$
                                        SET_VALUE=r.rectmethodmode[0],$
                                        UVALUE='Rectification Method')
            widget_control, w.rectmethod_bg,SENSITIVE=r.rectmethodmode[1]
            
            w.fixbdpx_bg = cw_bgroup(base,$
                                     FONT=buttonfont,$
                                     ['Fix Bad Pixels'],$
                                     /ROW,$
                                     /NONEXCLUSIVE,$
                                     SET_VALUE=r.fixbdpx[0],$
                                     UVALUE='Fix Bad Pixels')
            widget_control, w.fixbdpx_bg,SENSITIVE=r.fixbdpx[1]

         base = widget_base(row_base,$
                            /COLUMN,$
                            /BASE_ALIGN_LEFT,$
                            FRAME=2)

            label = widget_label(base,$
                                 VALUE='Wavelength Calibration:',$
                                 FONT=buttonfont,$
                                 /ALIGN_LEFT)
         
;            bg = cw_bgroup(base,$
;                           FONT=buttonfont,$
;                           ['Plot 1DXD Line Finding'],$
;                           /ROW,$
;                           /NONEXCLUSIVE,$
;                           /RETURN_NAME,$
;                           UVALUE='Plot 1DXD Line Finding',$
;                           SET_VALUE=0)
;
;            bg = cw_bgroup(base,$
;                           FONT=buttonfont,$
;                           ['Plot 2DXD Line Finding'],$
;                           /ROW,$
;                           /NONEXCLUSIVE,$
;                           /RETURN_NAME,$
;                           UVALUE='Plot 2DXD Line Finding',$
;                           SET_VALUE=0)

            bg = cw_bgroup(base,$
                           FONT=buttonfont,$
                           ['1D Line Find QA Plot'],$
                           /ROW,$
                           /NONEXCLUSIVE,$
                           /RETURN_NAME,$
                           UVALUE='1D Line Find QA Plot',$
                           SET_VALUE=0)

            w.loglin_bg = cw_bgroup(base,$
                                    LABEL_LEFT=' ',$
                                    FONT=buttonfont,$
                                    ['Log/Linear Plot'],$
                                    /ROW,$
                                    /NONEXCLUSIVE,$
                                    /RETURN_NAME,$
                                    UVALUE='Log/Linear Plot',$
                                    SET_VALUE=0)            
            widget_control, w.loglin_bg,SENSITIVE=0
            
            bg = cw_bgroup(base,$
                           FONT=buttonfont,$
                           ['2D Line Fits QA Plot'],$
                           /ROW,$
                           /NONEXCLUSIVE,$
                           /RETURN_NAME,$
                           UVALUE='2D Line Fits QA Plot',$
                           SET_VALUE=0)

            
            bg = cw_bgroup(base,$
                           FONT=buttonfont,$
                           ['2D Line Summary QA Plot'],$
                           /ROW,$
                           /NONEXCLUSIVE,$
                           /RETURN_NAME,$
                           UVALUE='2D Line Summary QA Plot',$
                           SET_VALUE=0)


         base = widget_base(row_base,$
                            /COLUMN,$
                            /BASE_ALIGN_LEFT,$
                            FRAME=2)            
            
            label = widget_label(base,$
                                 VALUE='Misc. Parameters:',$
                                 FONT=buttonfont,$
                                 /ALIGN_LEFT)

            fld = coyote_field2(base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Flat Field Offset:',$
                                UVALUE='Flat Field Offset',$
                                XSIZE=4,$
                                VALUE=0,$
                                TEXTID=textid)
            w.flatfieldoffset_fld = [fld,textid]
            
            fld = coyote_field2(base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Bad Pix Thresh:',$
                                UVALUE='Bad Pix Thresh',$
                                XSIZE=4,$
                                VALUE=instr.bdpxthresh,$
                                TEXTID=textid)
            w.bdpxthresh_fld = [fld,textid]
            
            row = widget_base(base,$
                              /ROW,$
                              /BASE_ALIGN_CENTER)
            
               fld = coyote_field2(row,$
                                   LABELFONT=buttonfont,$
                                   FIELDFONT=textfont,$
                                   TITLE='Ap Signs:',$
                                   UVALUE='PS Aperture Signs',$
                                   XSIZE=5,$
                                   TEXTID=textid)
               w.psapsigns_fld = [fld,textid]
               
               button = widget_button(row,$
                                      VALUE='Override',$
                                      UVALUE='Override Signs',$
                                      FONT=buttonfont)
               
;               fld = coyote_field2(base,$
;                                   LABELFONT=buttonfont,$
;                                   FIELDFONT=textfont,$
;                                   TITLE='Ybuffer:',$
;                                   UVALUE='Ybuffer',$
;                                   VALUE=instr.ybuffer,$
;                                   XSIZE=2,$
;                                   TEXTID=textid)
;               w.ybuffer_fld = [fld,textid]

            row = widget_base(base,$
                              /ROW)
            
;               fld = coyote_field2(base,$
;                                   LABELFONT=buttonfont,$
;                                   FIELDFONT=textfont,$
;                                   TITLE='Atmos Thresh:',$
;                                   UVALUE='Atmos Thresh',$
;                                   VALUE=instr.atmosthresh,$
;                                   XSIZE=5,$
;                                   TEXTID=textid)
;               w.atmosthresh_fld = [fld,textid]

               if instr.instrument eq 'TripleSpec (Palomar)' then begin

                  w.paltspecbias_bg = cw_bgroup(base,$
                                                FONT=buttonfont,$
                                                'Bias Correction',$
                                                /ROW,$
                                                /NONEXCLUSIVE,$
                                                SET_VALUE=r.paltspecbias,$
                                           UVALUE='PalTSpec Bias Correction')
                                
               endif

               
               
;         base = widget_base(row_base,$
;                            /COLUMN,$
;                            /BASE_ALIGN_RIGHT,$
;                            FRAME=2)
;         
;            label = widget_label(base,$
;                                 VALUE='Instrument Parameters:',$
;                                 FONT=buttonfont,$
;                                 /ALIGN_LEFT)

               
;  Eng Base
               
  w.eng_base = widget_base(work_base,$
                           /ROW,$
                           MAP=0)
  
      row_base = widget_base(w.eng_base,$
                             /ROW)

         col1_base = widget_base(row_base,$
                                 /COLUMN,$
                                 /BASE_ALIGN_RIGHT,$
                                 FRAME=2)

;            fld = coyote_field2(col1_base,$
;                                LABELFONT=buttonfont,$
;                                FIELDFONT=textfont,$
;                                TITLE='Oversampling:',$
;                                UVALUE='Oversampling',$
;                                XSIZE=5,$
;                                VALUE=instr.oversamp,$
;                                TEXTID=textid)
;            w.oversamp_fld = [fld,textid]

;            fld = coyote_field2(col1_base,$
;                                LABELFONT=buttonfont,$
;                                FIELDFONT=textfont,$
;                                TITLE='Mask Window:',$
;                                UVALUE='Mask Window',$
;                                XSIZE=5,$
;                                VALUE=10,$
;                                TEXTID=textid)
;            w.maskwindow_fld = [fld,textid]

;            button = widget_button(col1_base,$
;                                   FONT=buttonfont,$
;                                   VALUE='Make Order Images',$
;                                   UVALUE='Make Order Images')


         base = widget_base(row_base,$
                            /COLUMN,$
                            /BASE_ALIGN_RIGHT,$
                            FRAME=2)
         
            label = widget_label(base,$
                                 VALUE='Trace Parameters:',$
                                 FONT=buttonfont,$
                                 /ALIGN_LEFT)            

            
            fld = coyote_field2(base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Step Size (pixels):',$
                                UVALUE='Trace Step Size Field',$
                                VALUE=instr.tracestepsize,$
                                XSIZE=4,$
                                TEXTID=textid)
            w.tracestepsize_fld = [fld,textid]
                        
            fld = coyote_field2(base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Poly Deg:',$
                                UVALUE='Poly Fit Order Field',$
                                XSIZE=4,$
                                VALUE=instr.tracedeg,$
                                TEXTID=textid)
            w.tracedeg_fld = [fld,textid]
            
            
            fld = coyote_field2(base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Sumap (pixels):',$
                                UVALUE='Sumap Field',$
                                XSIZE=4,$
                                VALUE=instr.tracesumap,$
                                TEXTID=textid)
            w.sumap_fld = [fld,textid]
            
            fld = coyote_field2(base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Cen Thresh (pixels):',$
                                UVALUE='Win Trace Thresh Field',$
                                XSIZE=4,$
                                VALUE=instr.tracewinthresh,$
                                TEXTID=textid)
            w.tracewinthresh_fld = [fld,textid]
            
            fld = coyote_field2(base,$
                                LABELFONT=buttonfont,$
                                FIELDFONT=textfont,$
                                TITLE='Sig Thresh:',$
                                UVALUE='Sig Trace Thresh Field',$
                                XSIZE=4,$
                                VALUE=instr.tracesigthresh,$
                                TEXTID=textid)
            w.tracesigthresh_fld = [fld,textid]
            
         base = widget_base(row_base,$
                            /COLUMN,$
                            /BASE_ALIGN_RIGHT,$
                            FRAME=2)
         


            
; Get things running.  Center the widget using the Fanning routine.

cgcentertlb,w.xspextool_base

widget_control, w.xspextool_base, /REALIZE

; Start the Event Loop. This will be a non-blocking program.

XManager, 'xspextool', $
  w.xspextool_base, $
  /NO_BLOCK,$
  CLEANUP = 'xspextool_cleanup'

;  Load the three structures in the state structure.

  state = {w:w,r:r,d:d}

;  Set defaults

  xspextool_event,{ID:state.w.filereadmode_bg,TOP:state.w.xspextool_base,$
                   HANDLER:1L,SELECT:1,VALUE:instr.filereadmode}

  xspextool_event,{ID:state.w.reducmode_bg,TOP:state.w.xspextool_base,$
                   HANDLER:1L,SELECT:1,VALUE:instr.reductionmode}

  xspextool_event,{ID:state.w.combimgstat_dl,TOP:state.w.xspextool_base,$
                   HANDLER:1L,INDEX:state.r.combimgstat}

  xspextool_event,{ID:state.w.optext_bg,TOP:state.w.xspextool_base,$
                   HANDLER:1L,SELECT:state.r.optext[0],VALUE:state.r.optext[0]}

  xspextool_event,{ID:state.w.psbgsub_bg,TOP:state.w.xspextool_base,$
                   HANDLER:1L,SELECT:state.r.psbgsub,VALUE:state.r.psbgsub}

  xspextool_event,{ID:state.w.xsbgsub_bg,TOP:state.w.xspextool_base,$
                   HANDLER:1L,SELECT:state.r.xsbgsub,VALUE:state.r.xsbgsub}

  xspextool_event,{ID:state.w.medprofile_bg,TOP:state.w.xspextool_base,$
                   HANDLER:1L,SELECT:state.r.medprofile[0], $
                   VALUE:state.r.medprofile[0]}  

;  xspextool_event,{ID:state.w.autoxcor_bg,TOP:state.w.xspextool_base,$
;                   HANDLER:1L,SELECT:state.r.autoxcorr[0], $
;                   VALUE:state.r.autoxcorr[0]}

  xspextool_event,{ID:state.w.plotxcor_bg,TOP:state.w.xspextool_base,$
                   HANDLER:1L,SELECT:state.r.plotautoxcorr[0], $
                   VALUE:state.r.plotautoxcorr[0]}

  xspextool_loadpaths,/FROMFILE,CANCEL=cancel

;  Get bad pixel mask and invert 

  if strlowcase(state.r.bdpxmask) ne 'none' then begin

     xspextool_message, 'Loading bad pixel mask...' 
     
     bdpxmk = readfits(filepath(state.r.bdpxmask, $
                                ROOT=state.r.packagepath,SUBDIR='data'),/SILENT)
     *state.d.mask = byte(bdpxmk)
     
  endif else *state.d.mask = intarr(state.r.ncols,state.r.nrows)+1

  xspextool_message,'Setup complete.'
    
end
;
;******************************************************************************
;
; ------------------------------Event Handlers-------------------------------- 
;
;******************************************************************************
;

 pro xspextool_menuevent,event

  common xspextool_state

  widget_control, event.id,  GET_UVALUE = uvalue

  if uvalue eq 'Help' then begin

     pre = (strupcase(!version.os_family) eq 'WINDOWS') ? 'start ':'open '
     
     spawn, pre+filepath(strlowcase(state.w.instrument)+'_spextoolmanual.pdf', $
                         ROOT=state.r.packagepath,$
                         SUBDIR='manual')
     
     goto, cont
     
  endif

  if state.w.panel eq 'Paths' and uvalue ne 'Paths' then begin

     xspextool_loadpaths,/FROMPANEL,CANCEL=cancel
     if cancel then goto, cont
     
  endif

;  Blank out all of them

  widget_control, state.w.imageinput_base, MAP = 0
  widget_control, state.w.path_base,       MAP = 0
  widget_control, state.w.cals_base,       MAP = 0  
  widget_control, state.w.combimgs_base,   MAP = 0
  widget_control, state.w.ps_base,         MAP = 0
  widget_control, state.w.xs_base,         MAP = 0
  widget_control, state.w.other_base,      MAP = 0

  widget_control, state.w.oprefix_fld[0], SENSITIVE=0        
;  widget_control, state.w.iprefix_fld[0],  SENSITIVE=0        
  widget_control, state.w.outname_fld[0],   SENSITIVE=0        
  
  val = mc_cfld(state.w.iprefix_fld,7,CANCEL=cancel)
  if cancel then return
  if val ne 'flat' and val ne 'arc' then state.r.prefix = strtrim(val,2)
  
  case uvalue of 
            
     'Cals': begin
        
        widget_control, state.w.cals_base, /MAP
        widget_control, state.w.iprefix_fld[0],SENSITIVE=1
        
     end

     'Combine Images': begin
        
        widget_control, state.w.combimgs_base, /MAP
        if state.r.filereadmode eq 'Index' then $
           widget_control, state.w.iprefix_fld[0], /SENSITIVE
        widget_control, state.w.iprefix_fld[1],SET_VALUE=state.r.prefix
        
     end
     
     'Eng.': widget_control, state.w.eng_base, /MAP
     
     'Extended Source': begin
        
        state.w.base = 'Extended Source'
        widget_control, state.w.xs_base, /MAP
        widget_control, state.w.imageinput_base, /MAP
        if state.r.filereadmode eq 'Index' then begin
           
           widget_control, state.w.oprefix_fld[0], /SENSITIVE
           widget_control, state.w.iprefix_fld[0],  /SENSITIVE
           widget_control, state.w.iprefix_fld[1], SET_VALUE=state.r.prefix
           
        endif
        if state.r.filereadmode eq 'Filename' then $
           widget_control, state.w.outname_fld[0], /SENSITIVE
        
     end

     'Flat': begin
        
        widget_control, state.w.flat_base, /MAP
        if state.r.filereadmode eq 'Index' then $
           widget_control, state.w.iprefix_fld[0], /SENSITIVE
        
     end
     
     'Other': widget_control, state.w.other_base, /MAP
     
     'Paths': begin
        
        widget_control, state.w.path_base, /MAP
        if state.r.filereadmode eq 'Index' then $
           widget_control, state.w.iprefix_fld[0], /SENSITIVE
        
     end

     'Point Source': begin
        
        state.w.base = 'Point Source'
        widget_control, state.w.ps_base,         /MAP
        widget_control, state.w.imageinput_base, /MAP
        if state.r.filereadmode eq 'Index' then begin
           
           widget_control, state.w.oprefix_fld[0], /SENSITIVE
           widget_control, state.w.iprefix_fld[0], /SENSITIVE
           widget_control, state.w.iprefix_fld[1], SET_VALUE=state.r.prefix
           
        endif
        if state.r.filereadmode eq 'Filename' then $
           widget_control, state.w.outname_fld[0], /SENSITIVE
        
     end
     
  endcase


  state.w.panel = strtrim(uvalue,2)
  
cont:
  
end
;
;******************************************************************************
;
pro xspextool_event, event


  common xspextool_state

  widget_control, event.id,  GET_UVALUE = uvalue
  widget_control, /HOURGLASS
  
  case uvalue of
     
     'ABBA to ABAB': state.r.abba2abab = event.select
     
     'Amp Correction': state.r.ampcor[0] = event.select

;     'Auto X-Correlate': state.r.autoxcorr[0] = event.select

     'Cal Path Button': begin
        
        path =dialog_pickfile(/DIRECTORY,DIALOG_PARENT=state.w.xspextool_base,$
                              TITLE='Select Path',/MUST_EXIST)
        
        if path ne '' then widget_control,state.w.calpath_fld[1], $
                                          SET_VALUE = path
        
     end
     
;     'Check Seeing':state.r.checkseeing = event.select
     
     'Clear Data Path': widget_control, state.w.datapath_fld[1], SET_VALUE=''
     
     'Clear Proc Path': widget_control, state.w.procpath_fld[1], SET_VALUE=''

     'Clear Cal Path': widget_control, state.w.calpath_fld[1], SET_VALUE=''
     
     'Clear Table': widget_control, state.w.table,$
                                    SET_VALUE=strarr(2,state.r.tablesize)
     
     'Combine Images Mods': begin

        case event.value of 

           0:  state.r.combsclordrs=event.select
           1:  state.r.combbgsub=event.select

        endcase
        sum = state.r.combsclordrs+state.r.combbgsub 

        widget_control, state.w.combflat_base,SENSITIVE=sum

     end

     'Combine Images Button': begin

        xspextool_creadmode,cancel
        if cancel then return
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.datapath,$
                              FILTER=['*.fits;*.fits.gz'],/FIX_FILTER,$
                              /MULTIPLE_FILES)
        
        if path[0] ne '' then $
           widget_control,state.w.combimgs_fld[1],$
                          SET_VALUE=strjoin(file_basename(path),',',/SINGLE)
        
     end
     
     'Image Combination Statistic': begin
        
        state.r.combimgstat = event.index
        widget_control, state.w.combimgthresh_fld[0], $
                        SENSITIVE=(event.index le 2) ? 1:0
        
     end

     'Combine Images': xspextool_combineimgs
     
     'Combine Reduction Mode': begin

        state.r.combreductionmode = event.value
        if state.w.notirtf then $
           widget_control, state.w.abba2abab, MAP=(event.value eq 'A') ? 0:1

     end

     'Combine Flat Field Button': begin
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.calpath,$
                              FILTER='*.fits',/FIX_FILTER)
        
        if path ne '' then widget_control,state.w.combflat_fld[1], $
                                          SET_VALUE=file_basename(path)
 
     end
     
     'Data Path Button': begin
        
        path= dialog_pickfile(/DIRECTORY,DIALOG_PARENT=state.w.xspextool_base,$
                              TITLE='Select Path',/MUST_EXIST)
        
        if path ne '' then widget_control,state.w.datapath_fld[1], $
                                          SET_VALUE = path

     end
     
     'Show BG Region': xspextool_definebg

     'Determine Solution': begin
        
        state.r.detsol = event.select
        widget_control, state.w.wavebox_base, SENSITI=(event.select eq 1) ? 1:0
        
     end
     
     'Do All Steps': xspextool_doallsteps
     
     'Find Aperture Positions': xspextool_psapfind
     
     'Fix Bad Pixels': state.r.fixbdpx[0] = event.select
     
     'Flat Field': state.r.flatfield[0] = event.select
     
     'Full Wavecal Name Button': begin
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.calpath,$
                              FILTER='*.fits',/FIX_FILTER)
        
        if path ne '' then widget_control,state.w.wavecal_fld[1], $
                                          SET_VALUE=file_basename(path)
        
     end
     
     'Full Flat Name Button': begin
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.calpath,$
                              FILTER='*.fits',/FIX_FILTER)
        
        if path ne '' then widget_control,state.w.flatfield_fld[1], $
                                          SET_VALUE=file_basename(path)

     end

     'Image Combine Directory': state.r.combimgdir = event.value

     'Input Prefix Button': begin

        widget_control, state.w.iprefix_fld[1],SET_VALUE=event.value

     end
        
     'Linearity Correction': state.r.lc[0] = event.select
     
     'Load Single Image': xspextool_load1image,CANCEL=cancel

     'Load Trace File': xspextool_psapfind,CANCEL=cancel

     'Log/Linear Plot': state.r.mkloglinplot = event.select

     '1D Line Find QA Plot': begin

        state.r.mk1dlinefindplot = event.select
        widget_control, state.w.loglin_bg,SENSITIVE=event.select

   end
     
     '2D Line Fits QA Plot': state.r.mk2dlinefitsplot = event.select
     
     '2D Line Summary QA Plot': state.r.mklinesummaryplot = event.select
     
     'Make Spatial Profiles': xspextool_mkprofiles
     
     'Median Profile':  begin

       state.r.medprofile[0] = event.select
        widget_control, state.w.medprofile_bg,SET_VALUE=state.r.medprofile[0],$
                        SENSITIVE=state.r.medprofile[1]
       
    end
       
     'N Apertures': state.r.psnaps = event.index+1
     
     'Optimal Extraction': begin

        state.r.optext[0] = event.select
        widget_control, state.w.optext_bg,SET_VALUE=state.r.optext[0],$
                        SENSITIVE=state.r.optext[1]
        widget_control, state.w.psfradius_fld[1],SENSITIVE=state.r.optext[0]

        if state.r.optext[0] then begin
           
           state.r.psbgsub = 1
           widget_control, state.w.psbgsub_bg, SET_VALUE=1
           widget_control, state.w.psbginfo_base, SENSITIVE=1
           
        endif
                
     end

     'Override Signs': xspextool_overridesigns

     'PalTSpec Bias Correction': state.r.paltspecbias=event.select
     
     'Plot Auto X-Correlate': state.r.plotautoxcorr[0] = event.select
     
     'Plot Profiles': xspextool_updateappos,event
     
     'Plot Residuals':state.r.plotresiduals =event.select
     
     'Proc Path Button': begin
        
        path= dialog_pickfile(/DIRECTORY,DIALOG_PARENT=state.w.xspextool_base,$
                              TITLE='Select Path',/MUST_EXIST)
        
        if path ne '' then widget_control,state.w.procpath_fld[1], $
                                          SET_VALUE = path

     end
     
     'PS BG Subtraction': begin
        
        state.r.psbgsub = event.select
        widget_control, state.w.psbginfo_base,SENSITIVE=event.select
        if state.r.optext[0] then state.r.optext[0] = event.select
        
     end

     'PS N Apertures': state.r.psnaps = event.value+1
     
     'PS Show Apertures': xspextool_psdefineap

     'PS Extract Spectra': xspextool_psextractspec

     'PS Manual Find Mode': begin
 
        state.r.psapfindmanmode = event.value
        state.r.useappositions = (event.value eq 'Fix') ? 1:0

        widget_control, state.w.useappos, $
                        SET_VALUE=(event.value eq 'Fix') ? [1]:[0]

     end
     
     'PS Object Find Mode': begin
        
        widget_control,state.w.psapfindcur_base, MAP=0
        widget_control,state.w.psapfindtrace_base,  MAP=0
        state.r.psapfindmode = event.value

        if event.value eq 'Import Trace' then begin
           
           widget_control, state.w.psapfindtrace_base, /MAP 
           mc_setfocus, state.w.psitracefile_fld
           widget_control, state.w.pstrace_base, SENSITIVE=0
           
        endif else begin
           
           widget_control, state.w.psapfindcur_base, /MAP
           widget_control, state.w.pstrace_base, /SENSITIVE
                     
        endelse

     end

     'PS Trace Objects': xspextool_pstracespec

     'Rectification Method': state.r.rectmethodmode[0] = event.value
     
     'Quit': widget_control, event.top, /DESTROY

     'File Readmode': begin
        
        widget_control, state.w.oprefix_fld[0], SENSITIVE=0
        widget_control, state.w.iprefix_fld[0],  SENSITIVE=0
        widget_control, state.w.outname_fld[0],   SENSITIVE=0
        if event.value eq 'Filename' then begin
           
           state.r.filereadmode = event.value
           if state.w.base eq 'Point Source' or $
              state.w.base eq 'Extended Source' then $
                 widget_control, state.w.outname_fld[0],   SENSITIVE=1
           mc_setfocus,state.w.outname_fld
           
        endif else begin
           
           state.r.filereadmode = event.value
           if state.w.base eq 'Point Source' or $
              state.w.base eq 'Extended Source' then begin
              
              widget_control, state.w.oprefix_fld[0], SENSITIVE=1
              widget_control, state.w.iprefix_fld[0], SENSITIVE=1
              mc_setfocus,state.w.iprefix_fld
              
           endif else begin
              
              widget_control, state.w.iprefix_fld[0], SENSITIVE=1
              mc_setfocus,state.w.iprefix_fld
              
           endelse
           
        endelse
        
     end

     'Reduction Mode': begin
        
        state.r.reductionmode = event.value
        widget_control,state.w.skyimage_base, $
                       SENSITIVE=(event.value eq 'A-Sky/Dark') ? 1:0
        
     end

     'Save Orders': xspextool_saveimages
          
     'Sky Image Button': begin
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.calpath,$
                              FILTER='*.fits',/FIX_FILTER)
        
        if path ne '' then widget_control,state.w.skyimage_fld[1], $
                                          SET_VALUE =file_basename(path)

     end
     
     
     'Source Images Button': begin

        xspextool_creadmode,cancel
        if cancel then return

        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.sourcepath,$
                              FILTER=['*.fits;*.fits.gz'],/MULTIPLE_FILES)
        
        if path[0] ne '' then $
           widget_control,state.w.sourceimages_fld[1],$
                          SET_VALUE=strjoin(file_basename(path),',',/SINGLE)
        
     end

     'Source Images Directory': begin

        state.r.sourcedir = event.str
        case event.str of 

           'raw/': state.r.sourcepath = state.r.datapath

           'cal/': state.r.sourcepath = state.r.calpath

           'proc/': state.r.sourcepath = state.r.procpath

        endcase

     end

     'Subtract Background': xspextool_bgsubtraction

     'Trace File Button': begin
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.calpath,$
                              FILTER='*.dat',/FIX_FILTER)
        
        if path ne '' then widget_control,state.w.psitracefile_fld[1], $
                                          SET_VALUE=file_basename(path)

     end
     
     'XS Object Find Mode': begin

        case event.value of

           'Cursor': begin

              widget_control,state.w.xsapfindcur_base, MAP=1
              widget_control,state.w.xsapfindkey_base,  MAP=0

           end

           'Keyboard': begin
              
              widget_control,state.w.xsapfindcur_base, MAP=0
              widget_control,state.w.xsapfindkey_base,  MAP=1

           end
        
        endcase
        
        state.r.xsapfindmode = event.value

;        if event.value eq 'Cursor' then begin
;           
;           widget_control, state.w.psapfindtrace_base, /MAP 
;           mc_setfocus, state.w.psitracefile_fld
;           widget_control, state.w.pstrace_base, SENSITIVE=0
;           
;        endif else begin
;           
;           widget_control, state.w.psapfindcur_base, /MAP
;           widget_control, state.w.pstrace_base, /SENSITIVE
;                     
;        endelse

     end

     
     'XS BG Subtraction': begin
        
        state.r.xsbgsub = event.select
        widget_control, state.w.xsbg_fld[0],SENSITIVE=event.select
        widget_control, state.w.xsbgfitdeg_fld[0],SENSITIVE=event.select

     end
     
;     'Use Stored Solution': state.r.usestoredwc = event.select
     
     'Use Ap Positions': state.r.useappositions = event.select

     'XS Define Apertures': xspextool_xsdefineap
     
     'XS Extract Spectra': xspextool_xsextractspec
     
     'XS Identify Aperture Positions': xspextool_xsapfind
     
     'XS Trace Objects': xspextool_xstracespec
     
     else:
     
  end
  
  cont: 
  
end
;
;******************************************************************************
;
; ----------------------------Support procedures------------------------------ 
;
;******************************************************************************
;
;
pro xspextool_creadmode,cancel

  common xspextool_state

  if state.r.filereadmode eq 'Index' then begin

     message = [['It looks like you are trying to load ' + $
                 'data via a file name.'],$
                ['Please chnage the File Read Mode to "Filename" first.']]
     
     ok = dialog_message(message,/ERROR,DIALOG_PARENT=state.w.xspextool_base)
     cancel = 1

  endif else cancel = 0

end
;
;==============================================================================
;
pro xspextool_cleanup,xspextool_base

  common xspextool_state

if n_elements(state) ne 0 then begin

    ptr_free, state.r.appos
    ptr_free, state.r.apsign
    ptr_free, state.r.edgecoeffs
    ptr_free, state.r.guessappos
    ptr_free, state.r.hdrinfo
;    ptr_free, state.r.itot
    ptr_free, state.r.orders
    ptr_free, state.r.psdoorders
    ptr_free, state.r.rms
    ptr_free, state.r.tracecoeffs
    ptr_free, state.r.workfiles
    ptr_free, state.r.xranges
    ptr_free, state.r.xsdoorders
    ptr_free, state.r.profiles

    ptr_free, state.d.bdpxmask
    ptr_free, state.d.flat
    ptr_free, state.d.wavecal
    ptr_free, state.d.spatcal
;    ptr_free, state.d.xy2w
;    ptr_free, state.d.xy2s
    ptr_free, state.d.disp
    ptr_free, state.d.spectra
    ptr_free, state.d.varimage
    ptr_free, state.d.workimage
    ptr_free, state.d.omask
    ptr_free, state.d.mask
    ptr_free, state.d.bitmask
;    ptr_free, state.d.xo2w
;    ptr_free, state.d.tri
    ptr_free, state.d.imgstruc
    ptr_free, state.d.varstruc
    ptr_free, state.d.bpmstruc

    ptr_free, state.d.indices
    ptr_free, state.d.awave
    ptr_free, state.d.atrans
    ptr_free, state.r.axranges
    
endif
state = 0B

end
;
;==============================================================================
;
pro xspextool_combineimgs, CANCEL=cancel

  common xspextool_state

  cancel = 0

;  Get file readmode info.

  index    = (state.r.filereadmode eq 'Index') ? 1:0
  filename = (state.r.filereadmode eq 'Filename') ? 1:0
  if index then begin
     
     prefix = mc_cfld(state.w.iprefix_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     state.r.prefix = strtrim(prefix,2)
     
  endif
  
;  Get user inputs
  
  files = mc_cfld(state.w.combimgs_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return

  files = mc_fsextract(files,INDEX=index,FILENAME=filename,CANCEL=cancel)
  if cancel then return

  if index then begin

     result = mc_checkdups(state.r.datapath,files,state.r.nint,$
                           WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then return

  endif

  files = mc_mkfullpath(state.r.datapath,files,INDEX=index,FILENAME=filename,$
                        NI=state.r.nint,PREFIX=prefix,SUFFIX=state.r.suffix,$
                        WIDGET_ID=state.w.xspextool_base,/EXIST,CANCEL=cancel)
  if cancel then return
  nfiles = n_elements(files)
  
  oname = mc_cfld(state.w.comboname_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  pair = (state.r.combreductionmode eq 'A') ? 0:1

  if pair then begin
     
     if nfiles mod 2 then begin

        ok = dialog_message('Need an even number of images for A-B.', $
                            /ERROR,DIALOG_PARENT=state.w.xspextool_base)
        cancel = 1
        return     
        
     endif

     if state.w.notirtf then begin

        if state.r.abba2abab then begin

           idx = intarr(nfiles)
           for i = 0,nfiles/2-1 do begin
              
              if i mod 2 then begin
                 
                 idx[i*2] = i*2+1
                 idx[i*2+1] = i*2
                 
              endif else begin
                 
                 idx[i*2] = i*2
                 idx[i*2+1] = i*2+1
                 
              endelse
              
           endfor
           files = files[idx]
                      
        endif

     endif else files = mc_reorder(files,CANCEL=cancel)
     if cancel then return
     
  endif

  if state.r.combimgstat le 2 then begin
     
     thresh = mc_cfld(state.w.combimgthresh_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     
  endif

;  Read the flat if necessary

  if state.r.combsclordrs or state.r.combbgsub then begin

     flatname = mc_cfld(state.w.combflat_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     flatpath = mc_cfile(state.r.calpath+flatname,$
                         WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then return
     
     mc_readflat,flatpath,flat,ncols,nrows,modename,ps,slith_pix,slith_arc, $
                 slitw_pix,slitw_arc,rp,norders,orders,edgecoeffs,xranges, $
                 rms,rotation,OMASK=omask,CANCEL=cancel
     if cancel then return
     
  endif else rotation = 0
     
;  Check to see if it is PalTSPEC for bias correction.

     if state.r.instr eq 'TripleSpec (Palomar)' then begin

        extra={test:state.r.paltspecbias}
    
     endif else begin

        extra={test:state.r.ampcor[0]}

     endelse

;  kill ximgtool to save memory

     if xregistered('ximgtool') then ximgtool,/QUIT
     
;  Load Images

;  Load the data

     message = 'Loading the data'

     if state.r.lc[0] then begin

        message = message+' and correcting for non-linearity...'

     endif else message = message+'...'
     
     
  xspextool_message, message

  call_procedure,state.r.fitsreadprog,files,idata,hdrinfo,ivar, $
                 KEYWORD=state.r.keywords,NIMAGES=nimages,PAIR=pair, $
                 ROTATE=rotation,LINCORR=state.r.lc[0],$
                 BITINFO={lincormax:state.r.lincormax,lincormaxbit:0},$
                 BITMASK=bitmask,WIDGET_ID=state.w.xspextool_base, $
                 _EXTRA=extra,CANCEL=cancel
  if cancel then return

;  itot = (pair eq 1) ? total(itot)/2.:total(itot)

  if n_elements(files) gt 1 then begin
    
;  Scale Images if requested

     if state.r.combsclordrs then begin
        
        xspextool_message,'Scaling the images...'
        
        mc_scalespecsky,idata,ivar,omask,orders,/UPDATE,$
                        WIDGET_ID=state.w.xspextool_base,CANCEL=cancel
        if cancel then return
        
     endif
     
;  Subtract sky if requested.
     
     if state.r.combbgsub  then begin
        
        xspextool_message,'Subtracting sky...'
        
        mc_subspecsky,idata,ivar,edgecoeffs,norders,3,xranges,/UPDATE,$
                      WIDGET_ID=state.w.xspextool_base,CANCEL=cancel
        if cancel then return
        
     endif

     xspextool_message,'Combining the Images...'
     
     case state.r.combimgstat of
        
        0: begin
           
           mc_meancomb,idata,mean,mvar,DATAVAR=ivar,ROBUST=thresh, $
                       OGOODBAD=ogoodbad,/UPDATE, $
                       WIDGET_ID=state.w.xspextool_base,CANCEL=cancel
           if cancel then return
           
           bitmask = temporary(bitmask)*ogoodbad
           
        end
        
        1: begin
           
           mc_meancomb,idata,mean,mvar,/RMS,ROBUST=thresh, $
                       OGOODBAD=ogoodbad,/UPDATE,$
                       WIDGET_ID=state.w.xspextool_base,CANCEL=cancel
           if cancel then return
           
           bitmask = temporary(bitmask)*ogoodbad
           
        end
        
        2: begin
           
           mc_meancomb,idata,mean,mvar,ROBUST=thresh,OGOODBAD=ogoodbad,$
                       /UPDATE,WIDGET_ID=state.w.xspextool_base,CANCEL=cancel
           if cancel then return
           
           bitmask = temporary(bitmask)*ogoodbad
           
        end
        
        3: begin
           
           mc_meancomb,idata,mean,mvar,DATAVAR=ivcube,CANCEL=cancel
           if cancel then return
           
        end
        
        4: begin
           
           mc_meancomb,idata,mean,mvar,/RMS,CANCEL=cancel
           if cancel then return
           
        end
        
        5: begin
           
           mc_meancomb,idata,mean,mvar,CANCEL=cancel
           if cancel then return
           
        end
        
        6: begin
           
           mc_medcomb,idata,mean,mvar,/MAD,CANCEL=cancel
           if cancel then return
           
        end
        
        7: begin
           
           mc_medcomb,idata,mean,mvar,CANCEL=cancel
           if cancel then return
           
        end
        
     endcase
     
;  Combine bad pixel masks
     
     bitmask = mc_combflagstack(bitmask,CANCEL=cancel)
     if cancel then return

  endif else begin

     mean = reform(idata)
     mvar = reform(ivar)

  endelse
     
;  Write the results to disk

  avehdr = mc_avehdrs(hdrinfo,PAIR=pair,CANCEL=cancel)
  if cancel then return

  history = 'This image was created by '+ $
            state.r.combstats[state.r.combimgstat]+$
            ' combining the images '+strjoin(file_basename(files), ', ')+'.'
  
  if state.r.lc[0] then history = history+'  The raw data were '+$
                                  'corrected for non-linearity.'
  
  if state.r.combsclordrs then begin
     
     history = history+'  Each order within an image was first scaled to ' + $
               'the median flux value of all of the pixels in that order ' + $
               'in all of the images.'
     
  endif

  if state.r.combbgsub then begin
     
     history=history+'  The median flux level was subtracted off on a ' + $
             'column by column basis within each order.' 
     
  endif 

  avehdr.vals.filename = oname+'.fits'

;  Write out combined image

  fxhmake,hdr,mc_unrotate(mean,rotation,CANCEL=cancel)
  ntags = n_tags(avehdr.vals)
  names = tag_names(avehdr.vals)

  for i = 0,n_tags(avehdr.vals)-1 do begin
     
     if size(avehdr.vals.(i),/TYPE) eq 7 and $
        strlen(avehdr.vals.(i)) gt 68 then begin

        fxaddpar,hdr,names[i],avehdr.vals.(i),avehdr.coms.(i)
        
     endif else sxaddpar,hdr,names[i],avehdr.vals.(i),avehdr.coms.(i)

  endfor

  if state.r.keywords[0] ne '' then before=state.r.keywords[0]



  sxaddpar,hdr,'CREPROG','xspextool', ' Creation program',BEFORE=before
  sxaddpar,hdr,'VERSION',state.w.version, ' Spextool version',BEFORE=before
  sxaddpar,hdr,'AMPCOR', state.r.ampcor[0], ' Amplifier correction (bit flag)',$
           BEFORE=before
  sxaddpar,hdr,'LINCOR', state.r.lc[0], ' Linearity correction (bit flag)',$
           BEFORE=before
  sxaddpar,hdr,'FLATED', 0, ' Flat fielded (bit flag)',BEFORE=before
  sxaddpar,hdr,'NIMCOMB',nfiles,' Number of images combined',BEFORE=before
  sxaddpar,hdr,'IMCOMBAB',pair,' Pair subtraction?: 1=yes, 0=no',BEFORE=before
  sxaddpar,hdr,'LINCRMAX',state.r.lincormax, $
           ' Linearity correction maximum (DN)',BEFORE=before

  sxaddhist,' ',hdr
  sxaddhist,'################# Xspextool Combine Images History ' + $
            '################',hdr
  sxaddhist,' ',hdr
  
  history = mc_splittext(history,67,CANCEL=cancel)
  if cancel then return
  sxaddhist,history,hdr

  opath = (state.r.combimgdir eq 'proc/') ? state.r.procpath:state.r.calpath
 

  mwrfits,input,opath+oname+'.fits',hdr,/CREATE

;  Write combined image to first extension

  writefits,opath+oname+'.fits',mc_unrotate(mean,rotation),/APPEND

;  Write variance image to second extension

  writefits,opath+oname+'.fits',mc_unrotate(mvar,rotation),/APPEND

;  Write bitmask image to second extension

  writefits,opath+oname+'.fits',mc_unrotate(byte(bitmask),rotation),/APPEND


;  Update message and display image.
  
  xspextool_message,[[['Wrote combined images to ']], $
                     [[state.r.procpath+oname+'.fits']],$
                     [['**NOTE:  THIS PROCESS DID NOT FLAT-FIELD THE IMAGE**']]]

  ximgtool,mc_unrotate(mean/sqrt(mvar),rotation),BUFFER=3,RANGE='98%', $
           GROUP_LEADER=state.w.xspextool_base, $
           BITMASK={bitmask:mc_unrotate(bitmask,rotation),plot1:[0,2]},$
           STDIMAGE=state.r.stdimage,PLOTWINSIZE=state.r.plotwinsize, $
           FILENAME='S/N',POSITION=[1,0],CLEARFRAMES=[3,4,5],/NODISPLAY,/ZTOFIT
  
  ximgtool,sqrt(mc_unrotate(mvar,rotation)),BUFFER=2,RANGE='98%', $
           GROUP_LEADER=state.w.xspextool_base, $
           BITMASK={bitmask:mc_unrotate(bitmask,rotation),plot1:[0,2]},$
           STDIMAGE=state.r.stdimage,PLOTWINSIZE=state.r.plotwinsize, $
           ZUNITS='(DN / s)',FILENAME='Uncertainty',$
           POSITION=[1,0],/NODISPLAY,/ZTOFIT
  
  ximgtool,mc_unrotate(mean,rotation),BUFFER=1,RANGE='98%', $
           GROUP_LEADER=state.w.xspextool_base, $
           BITMASK={bitmask:mc_unrotate(bitmask,rotation),plot1:[0,2]},$
           STDIMAGE=state.r.stdimage,PLOTWINSIZE=state.r.plotwinsize, $
           ZUNITS='(DN / s)',FILENAME='Flux',/LOCK,/ZTOFIT
    
  xspextool_message,'Task complete.',/WINONLY
  
end
;
;******************************************************************************
;
pro xspextool_doallsteps

  common xspextool_state

  index    = (state.r.filereadmode eq 'Index') ? 1:0
  filename = (state.r.filereadmode eq 'Filename') ? 1:0
  
;  Get user inputs.
  
  files = mc_cfld(state.w.sourceimages_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  files = mc_fsextract(files,INDEX=index,FILENAME=filename,CANCEL=cancel)
  if cancel then return
  
  if state.r.reductionmode eq 'A-B' then begin
     
     if n_elements(files) mod 2 ne 0 then begin
        
        mess = 'Must be an even number of images for pair subtraction.'
        ok = dialog_message(mess,/ERROR,DIALOG_PARENT=state.w.xspextool_base)
        mc_setfocus,state.w.sourceimages_fld
        cancel = 1
        return
        
     endif
     
  endif
  
;  Get loop size
  
  loopsize = (state.r.reductionmode eq 'A-B') ? (n_elements(files)/2.):$
             (n_elements(files))
  
  for i = 0, loopsize-1 do begin
     
     subset = (state.r.reductionmode eq 'A-B') ? files[i*2:i*2+1]:files[i]
     
     *state.r.workfiles = subset
     xspextool_loadimages,/JUSTIMGS,CANCEL=cancel    
     if cancel then break
     xspextool_mkprofiles,/DOALLSTEPS,CANCEL=cancel
     if cancel then break

     if state.w.base eq 'Point Source' then begin
        
        xspextool_psapfind,/DOALLSTEPS,CANCEL=cancel
        if cancel then break        
        if state.r.psapfindmode eq 'Cursor' then $
           xspextool_pstracespec,CANCEL=cancel
        if cancel then break
        xspextool_psdefineap,CANCEL=cancel
        if cancel then break
        state.r.doallsteps = 1
        xspextool_psextractspec,CANCEL=cancel
        state.r.doallsteps = 0
        if cancel then break
        
        
     endif
     if state.w.base eq 'Extended Source' then begin
        
        xspextool_xsapfind,/DOALLSTEPS,CANCEL=cancel
        if cancel then break
        xspextool_xstracespec,CANCEL=cancel
        if cancel then break
        xspextool_xsdefineap,CANCEL=cancel
        if cancel then break
        state.r.doallsteps = 1
        xspextool_xsextractspec,CANCEL=cancel
        state.r.doallsteps = 0
        if cancel then break
        
     endif
     
  endfor

  xspextool_message,'Do All Steps Complete.'
  
  beep
  beep
  beep
  
end
;
;===============================================================================
;
pro xspextool_load1image,CANCEL=cancel

  common xspextool_state

  cancel = 0
  
;  File Read Mode.
  
  index    = (state.r.filereadmode eq 'Index') ? 1:0
  filename = (state.r.filereadmode eq 'Filename') ? 1:0
  
;  Get user inputs.
  
  files = mc_cfld(state.w.sourceimages_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  files = mc_fsextract(files,INDEX=index,FILENAME=filename, $
                       WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
  if cancel then return
  
  case state.r.reductionmode of 
     
     'A': begin
        
        if n_elements(files) ne 1 then begin
           
           mess = 'Can only load 1 image at a time in A mode.'
           ok = dialog_message(mess,/ERROR, $
                               DIALOG_PARENT=state.w.xspextool_base)
           mc_setfocus,state.w.sourceimages_fld
           cancel = 1
           return
           
        endif
        
     end

     'A-B': begin
        
        if n_elements(files) ne 2 then begin
           
           mess = 'Can only load 2 images at a time in A-B mode.'
           ok = dialog_message(mess,/ERROR, $
                               DIALOG_PARENT=state.w.xspextool_base)
           mc_setfocus,state.w.sourceimages_fld
           cancel = 1
           return
           
        endif
        
     end
     
     'A-Sky/Dark': begin
        
        if n_elements(files) ne 1 then begin
           
           mess = 'Can only load 1 image at a time in A-Sky/Dark mode.'
           ok = dialog_message(mess,/ERROR, $
                               DIALOG_PARENT=state.w.xspextool_base)
           mc_setfocus,state.w.sourceimages_fld
           cancel = 1
           return
           
        endif
                      
     end

 endcase

  *state.r.workfiles = files
  xspextool_loadimages,CANCEL=cancel
  if cancel then return

end
;
;===============================================================================
;
pro xspextool_loadimages,JUSTIMGS=justimgs,CANCEL=cancel

  common xspextool_state

  cancel = 0

;  File Read Mode.

  index    = (state.r.filereadmode eq 'Index') ? 1:0
  filename = (state.r.filereadmode eq 'Filename') ? 1:0
  if index then begin
     
     prefix = mc_cfld(state.w.iprefix_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     state.r.prefix = strtrim(prefix,2)

;  Check for duplicate files

     result = mc_checkdups(state.r.sourcepath,*state.r.workfiles, $
                           state.r.nint,WIDGET_ID=state.w.xspextool_base, $
                           CANCEL=cancel)
     if cancel then return
     
  endif
  
;  Get full paths

  fullpaths = mc_mkfullpath(state.r.sourcepath,*state.r.workfiles,INDEX=index,$
                            FILENAME=filename,NI=state.r.nint,PREFIX=prefix,$
                            SUFFIX=state.r.suffix, $
                            WIDGET_ID=state.w.xspextool_base,/EXIST, $
                            CANCEL=cancel)
  if cancel then mc_setfocus,state.w.sourceimages_fld
  if cancel then return

;  Re order the files for extended source to keep them ABABAB.

  if state.r.reductionmode eq 'A-B' and $
;     state.w.base eq 'Extended Source' and $
     ~state.w.notirtf then begin    

     fullpaths = mc_reorder(fullpaths,IDX=idx,CANCEL=cancel)
     if cancel then return     
     *state.r.workfiles = (*state.r.workfiles)[idx]
     
  endif

;  Get the name of the sky frame 
  
  if state.r.reductionmode eq 'A-Sky/Dark' then begin
     
     skyname = mc_cfld(state.w.skyimage_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     skypath = mc_cfile(state.r.calpath+skyname, $
                        WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then return
     
  endif

;  Upate the title of the Spextool window
  
  title = 'Spextool '+state.w.version+' for '+state.r.instr+' - '+ $
        strjoin(strtrim(*state.r.workfiles,2),', ')
  widget_control, state.w.xspextool_base,BASE_SET_TITLE=title

  
; Start loading the data 
  
  if ~keyword_set(JUSTIMGS) then begin

;  If this is NOT a Do All Steps call, then you have to load everything
     
     flatname = mc_cfld(state.w.flatfield_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     flatpath = mc_cfile(state.r.calpath+flatname, $
                         WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then mc_setfocus,state.w.flatfield_fld
     if cancel then return
       
     wavecal = mc_cfld(state.w.wavecal_fld,7,CANCEL=cancel)
     if cancel then return
     state.r.wavecal = (strtrim(wavecal,2) eq '') ? 0:1
     
     if state.r.wavecal then begin
        
        wavecalpath = mc_cfile(state.r.calpath+wavecal,$
                               WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then return        
        
     endif 

;  Now read flat field and load info.

     xspextool_message, 'Loading Flat Field...'
     mc_readflat,flatpath,flat,ncols,nrows,modename,ps,slith_pix,slith_arc, $
                 slitw_pix,slitw_arc,rp,norders,orders,edgecoeffs,xranges, $
                 rms,rotation,OMASK=omask,CANCEL=cancel
     if cancel then return
     
; Load necessary info into memory and check if obsmode has changed.
     
     *state.d.flat       = flat
     *state.d.omask      = omask
     *state.r.edgecoeffs = edgecoeffs
     *state.r.orders     = orders
     state.r.norders     = norders
     state.r.ps          = ps
     state.r.slith_arc   = slith_arc
     state.r.slitw_arc   = slitw_arc
     state.r.slith_pix   = slith_pix
     state.r.slitw_pix   = slitw_pix
     state.r.rp          = rp
     state.r.modename    = modename
     *state.r.xranges    = xranges
     state.r.rotation    = rotation
     state.r.ncols       = ncols
     state.r.nrows       = nrows
     *state.r.psdoorders = intarr(norders)+1
     *state.r.xsdoorders = intarr(norders)+1
     *state.d.bdpxmask   = rotate(*state.d.mask,state.r.rotation)
     
;  Change the obsmode if necessary.
     
     if state.r.obsmode ne modename then begin

        *state.r.psdoorders = intarr(state.r.norders)+1
        *state.r.xsdoorders = intarr(state.r.norders)+1
        
;  Now check mode for wavelength solution
        
        if modename eq 'LowRes15' or modename eq 'Prism' then begin
           
           state.r.usestoredwc = 1
           
        endif else state.r.usestoredwc = 0
        
        state.r.obsmode = modename
        
     endif

;  Load the wavecal file
     
     xspextool_message, 'Loading Wavelength Calibration File...'
     
     if state.r.wavecal then begin

        mc_readwavecal,wavecalpath,wctype,wcflatname,indices,wavecal, $
                       spatcal,xranges,wdisp,wavefmt,spatfmt, $
                       ROTATE=state.r.rotation,CANCEL=cancel
        if cancel then return

        state.r.wavefmt = wavefmt
        state.r.spatfmt = spatfmt
        *state.d.disp   = wdisp
        state.r.wctype  = wctype
        *state.d.indices = indices
        *state.r.axranges = xranges
        
;  Load the atmospheric transmission

;  Get the resolving powers available in Spextool/data/

        files = file_search(filepath('atran*.fits', $
                                     ROOT_DIR=state.r.spextoolpath, $
                                     SUBDIR='data'),COUNT=nfiles)
        rps = lonarr(nfiles)
        for i =0,nfiles-1 do $
           rps[i] = long(strmid(file_basename(files[i],'.fits'),5))        
       
;  Now find the closest one and load the atmospheric transmission
        
        min = min(abs(rps-rp),idx)
        
        spec = readfits(filepath('atran'+strtrim(rps[idx],2)+'.fits', $
                                 ROOT_DIR=state.r.spextoolpath,SUBDIR='data'))
        *state.d.awave = reform(spec[*,0])
        *state.d.atrans = reform(spec[*,1])

     endif else begin
        
;  If no wavecal file, then load in a fake one.

        mc_simwavecal2d,state.r.ncols,state.r.nrows,*state.r.edgecoeffs, $
                        *state.r.xranges,state.r.slith_arc,ps,wavecal,spatcal, $
                        indices,CANCEL=cancel
        if cancel then return

        state.r.wavefmt = '(I4.4)'
        state.r.spatfmt = '(f5.2)'
        *state.d.disp   = replicate(1.0,norders)
        state.r.wctype  = '1D'
        *state.d.indices = indices
        *state.r.axranges = *state.r.xranges
        
     endelse
     
  endif
  
;  Check to see if it is PalTSPEC for bias correction.
  
  if state.r.instr eq 'TripleSpec (Palomar)' then begin
     
     extra={test:state.r.paltspecbias}
     
  endif else begin
     
     extra={test:state.r.ampcor[0]}
     
  endelse
  
;  Load the data
  
  message = 'Loading the data'
  lcadd = (state.r.lc[0] eq 1) ? ' and correcting for non-linearity':''
  
  case state.r.reductionmode of 
     
     'A': begin
        
;  Check to see if the data are raw or processed.

        if state.r.sourcedir eq 'raw/' then begin
           
           xspextool_message, message+lcadd+'...'

           call_procedure,state.r.fitsreadprog,fullpaths,data,hdrinfo,var, $
                          KEYWORD=state.r.keywords,ROTATE=state.r.rotation, $
                          LINCORR=state.r.lc[0], $
                          BITINFO={lincormax:state.r.lincormax,lincormaxbit:0},$
                          BITMASK=bitmask,_EXTRA=extra,CANCEL=cancel
           if cancel then return
           
        endif else begin

           xspextool_message, message+'...'
           
           junk    = mrdfits(fullpaths,0,hdr,/SILENT)
           hdrinfo = mc_gethdrinfo(hdr,CANCEL=cancel)
           
           data = rotate(mrdfits(fullpaths,1,/SILENT),state.r.rotation)
           var  = rotate(mrdfits(fullpaths,2,/SILENT),state.r.rotation)
           bitmask = rotate(mrdfits(fullpaths,3,/SILENT),state.r.rotation)           
        endelse          
        
     end
     
     'A-B': begin

        xspextool_message, message+lcadd+'...'
        
        call_procedure,state.r.fitsreadprog,fullpaths,data,hdrinfo,var, $
                       KEYWORD=state.r.keywords,ROTATE=state.r.rotation, $
                       LINCORR=state.r.lc[0],/PAIR,$
                       BITINFO={lincormax:state.r.lincormax,lincormaxbit:0},$
                       BITMASK=bitmask,_EXTRA=extra,CANCEl=cancel
        if cancel then return
        
     end
     
     'A-Sky/Dark': begin

;  Check to see if the data are raw or processed.

        if state.r.sourcedir eq 'raw/' then begin

           xspextool_message, message+lcadd+'...'
           
           call_procedure,state.r.fitsreadprog,fullpaths,data,hdrinfo,var, $
                          KEYWORD=state.r.keywords, ROTATE=state.r.rotation, $
                          LINCORR=state.r.lc[0],$
                          BITINFO={lincormax:state.r.lincormax,lincormaxbit:0},$
                          BITMASK=bitmask,ITOT=itot,_EXTRA=extra,CANCEL=cancel
           if cancel then return
           
        endif else begin

           xspextool_message, message+'...'
           
           hdr = headfits(fullpaths[0],EXTEN=0)
           hdrinfo = mc_gethdrinfo(hdr,CANCEL=cancel)
           if cancel then return
           
           data = rotate(mrdfits(fullpaths[0],1,/SILENT),state.r.rotation)
           var = rotate(mrdfits(fullpaths[0],2,/SILENT),state.r.rotation)
           bitmask = rotate(mrdfits(fullpaths[0],3,/SILENT),state.r.rotation)
           
        endelse

        sky = rotate(mrdfits(skypath,1,/SILENT),state.r.rotation)
        skyvar = rotate(mrdfits(skypath,2,/SILENT),state.r.rotation)
        skybitmask = rotate(mrdfits(skypath,3,/SILENT),state.r.rotation)

;        ximgtool,data,sky,data-sky,/LOCK
;        return
        
        data = temporary(data)-sky
        var  = temporary(var)+skyvar      
        bitmask = mc_combflagstack([[[bitmask]],[[skybitmask]]],CANCEL=cancel)
        if cancel then return

     end

  endcase
  
;  Flat field the data
  
  if state.r.flatfield[0] then begin
     
     xspextool_message,'Flat-fielding the data...'
     
     data = temporary(data)/*state.d.flat
     var  = temporary(var)/(*state.d.flat)^2
     
  endif else xspextool_message,'Data not flat-fielded...'

;  Fix bad pixels for iSHELL cals or you have problems with NaNs in the spectra
  
;  htpxmk = rotate(readfits(filepath('ishell_htpxmk.fits', $
;                                    ROOT_DIR=state.r.packagepath, $
;                                    SUBDIR='data'),/SILENT),state.r.rotation)
;  
;  bdpxmk = rotate(readfits(filepath('ishell_bdpxmk.fits', $
;                                    ROOT_DIR=state.r.packagepath, $
;                                    SUBDIR='data'),/SILENT),state.r.rotation)
;
;  fixpix,data,htpxmk*bdpxmk,odata,/SILENT
;  data = temporary(odata)  
  
;  Rectifying the orders

  xspextool_message,'Rectifying the data...'

  if state.r.rectmethodmode[0] then begin

     progressbar = obj_new('SHOWPROGRESS',state.w.xspextool_base,COLOR=2,$
                           CANCELBUTTON=1,$
                           MESSAGE='Rectifying the orders')
     progressbar -> start
     
  endif

  for i = 0,state.r.norders-1 do begin

     st = mc_rectifyorder(data,(*state.d.indices).(i), $
                          state.r.rectmethodmode[0],VAR=var, $
                          BPMASK=rotate(*state.d.mask,state.r.rotation), $
                          BSMASK=bitmask,CANCEL=cancel)
     if cancel then return

     s = size(st.img,/DIMEN)
     
     timg = dblarr(s[0]+1,s[1]+1,/NOZERO)
     tvar = dblarr(s[0]+1,s[1]+1,/NOZERO)
     tbpm = dblarr(s[0]+1,s[1]+1,/NOZERO)
     tbsm = dblarr(s[0]+1,s[1]+1,/NOZERO)
     
     timg[1:*,0] = st.wgrid
     timg[0,1:*] = st.sgrid
     timg[1:*,1:*] = st.img

     tvar[1:*,0] = st.wgrid
     tvar[0,1:*] = st.sgrid
     tvar[1:*,1:*] = st.var
     
     tbpm[1:*,0] = st.wgrid
     tbpm[0,1:*] = st.sgrid
     tbpm[1:*,1:*] = st.bpm

     tbsm[1:*,0] = st.wgrid
     tbsm[0,1:*] = st.sgrid
     tbsm[1:*,1:*] = st.bsm

     tag = 'Order'+string((*state.r.orders)[i],FORMAT='(I3.3)')

     if i eq 0 then begin

        imgstruc = create_struct(tag,timg)
        varstruc = create_struct(tag,tvar)
        bpmstruc = create_struct(tag,tbpm)
        bsmstruc = create_struct(tag,tbsm)

     endif else begin

        imgstruc = create_struct(imgstruc,tag,timg)
        varstruc = create_struct(varstruc,tag,tvar)
        bpmstruc = create_struct(bpmstruc,tag,tbpm)
        bsmstruc = create_struct(bsmstruc,tag,tbsm)

     endelse

     if state.r.rectmethodmode[0] then begin
        
        cancel = progressBar->CheckCancel()
        if cancel then begin
           
           progressBar->Destroy
           obj_destroy, progressbar
           return
           
        endif
        percent = (i+1)*(100./float(norders))
        progressbar->update, percent
        
     endif
          
  endfor

  if state.r.rectmethodmode[0] then begin

    progressbar-> destroy
    obj_destroy, progressbar

  endif

  if state.r.wctype eq '2D' or state.r.wctype eq '2DXD' then begin

;  Stack the results

     stack = mc_stackrectorders(imgstruc,*state.r.orders, $
                                state.r.slith_arc,30,VARS=varstruc,$
                                BPMASKS=bpmstruc,BSMASKS=bsmstruc,CANCEL=cancel)
     if cancel then return

     *state.d.workimage  = stack.img
     *state.d.varimage   = stack.var
     *state.d.bdpxmask   = stack.bpm
     *state.d.bitmask    = stack.bsm
     *state.d.omask      = stack.omask
     *state.r.edgecoeffs = stack.edgecoeffs
     *state.r.xranges    = stack.xranges

;  Adjust the xranges if polyclipped
     
     if state.r.rectmethodmode[0] then *state.r.xranges = *state.r.xranges+0.5

     wavecal = stack.wavecal
     spatcal = stack.spatcal
     
  endif else begin
     
     *state.d.workimage = data
     *state.d.varimage  = var
     *state.d.bitmask   = bitmask
     *state.d.bdpxmask  = rotate(*state.d.mask,state.r.rotation)
     
  endelse

  delvarx, flat  
  delvarx, data
  delvarx, var
       
  *state.d.imgstruc = imgstruc

  if ~keyword_set(JUSTIMGS) then begin
  
     *state.d.wavecal = wavecal
     *state.d.spatcal = spatcal

  endif
  *state.r.hdrinfo = hdrinfo

;  Display image

  xspextool_ximgtool,/FORCE,CANCEL=cancel
  if cancel then return

  state.r.pscontinue = 1
  state.r.xscontinue = 1
  
  xspextool_message,'Task complete.',/WINONLY
  
end
;
;==============================================================================
;
pro xspextool_loadpaths,FROMFILE=fromfile,FROMPANEL=frompanel,CANCEL=cancel

  cancel = 0

  common xspextool_state

  widget_control, /HOURGLASS

  xspextool_message, 'Updating paths...' 

  if keyword_set(FROMFILE) then begin

;  Check to see if the path file exists.
    
     file = mc_cfile(state.r.pathsfile,/SILENT,CANCEL=cancel)
     if not cancel then begin
        
        paths = strarr(3)
        openr, lun, file,/GET_LUN
        readf, lun, paths
        free_lun, lun

        datapath = paths[0]
        calpath  = paths[1]
        procpath = paths[2]
        
     endif else begin

        cancel = 1
        return

     endelse

  endif

  if keyword_set(FROMPANEL) then begin

     datapath = mc_cfld(state.w.datapath_fld,7,CANCEL=cancel)
     calpath = mc_cfld(state.w.calpath_fld,7,CANCEL=cancel)
     procpath = mc_cfld(state.w.procpath_fld,7,CANCEL=cancel)
     
  endif

;  Check whether the paths exist or not.
        
  datapath = mc_cpath(datapath,CANCEL=cancel)

  if cancel then begin
     
     ok = dialog_message('Data path does not exist.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     mc_setfocus, state.w.datapath_fld
     widget_control, state.w.path_but, /SET_BUTTON
     cancel = 1
     return

  endif

  calpath = mc_cpath(calpath,CANCEL=cancel)
  if cancel then begin
        
     ok = dialog_message('Cal path does not exist.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     mc_setfocus, state.w.calpath_fld
     widget_control, state.w.path_but, /SET_BUTTON
     cancel = 1
     return
     
  endif

  procpath = mc_cpath(procpath,CANCEL=cancel)
  if cancel then begin
     
     ok = dialog_message('Proc path does not exist.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     mc_setfocus,state.w.procpath_fld
     widget_control, state.w.path_but, /SET_BUTTON
     cancel = 1
     return
     
  endif

;  Make sure they are unique paths
  
  if datapath eq calpath or datapath eq procpath or $
     calpath eq procpath then begin
     
     ok = dialog_message('The raw/, cal/, and proc/ paths must be ' + $
                         'different diretories.',/ERROR, $
                         DIALOG_PARENT=state.w.xspextool_base)
     widget_control, state.w.path_but, /SET_BUTTON
     cancel = 1
     return
     
  endif

  if datapath ne state.r.datapath then begin

     prefix = mc_findprefixes(datapath,state.r.nint,state.r.nsuffix,$
                              WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then begin

        cancel = 1
        return

     endif
     
     widget_control,state.w.datapath_fld[1],SET_VALUE=datapath
     mc_setfocus,state.w.datapath_fld
     state.r.datapath = datapath

     desc = ['1\Input Prefix','0\'+prefix]
     
     widget_control, state.w.iprefix_row,UPDATE=0
     widget_control, state.w.iprefix_but,/DESTROY
     
     state.w.iprefix_but = CW_PDMENU(state.w.iprefix_row, desc, $
                                     UVALUE='Input Prefix Button',$
                                     FONT=state.w.buttonfont,$
                                     /RETURN_NAME)
     
     z = where(prefix ne 'flat-' and prefix ne 'arc-')
     
     widget_control, state.w.iprefix_fld[1],SET_VALUE=prefix[z[0]]
     widget_control, state.w.iprefix_row,UPDATE=1
    
  endif
    
  if calpath ne state.r.calpath then begin
     
     widget_control,state.w.calpath_fld[1],SET_VALUE=calpath
     mc_setfocus,state.w.calpath_fld
     state.r.calpath  = calpath
     
  endif
  
  if procpath ne state.r.procpath then begin
     
     widget_control,state.w.procpath_fld[1],SET_VALUE=procpath
     mc_setfocus,state.w.procpath_fld
     state.r.procpath = procpath
     
  endif

  case state.r.sourcedir of 
     
     'raw/': state.r.sourcepath = state.r.datapath
     
     'cal/': state.r.sourcepath = state.r.calpath
     
     'proc/': state.r.sourcepath = state.r.procpath
     
  endcase

;  Write to disk
  
  openw, lun, state.r.pathsfile,/GET_LUN
  printf, lun, state.r.datapath
  printf, lun, state.r.calpath
  printf, lun, state.r.procpath
  free_lun, lun

  xspextool_message, 'Updating paths complete.' 
  
end
;
;==============================================================================
;
pro xspextool_message,message,WINONLY=winonly,SPACE=space

  common xspextool_state

  widget_control,state.w.messwin, SET_VALUE=message
  
  if not keyword_set(WINONLY) then begin

     print, message
     if keyword_set(SPACE) then print

  endif

end
;
;==============================================================================
;
pro xspextool_mkprofiles,DOALLSTEPS=doallsteps,CANCEL=cancel

  common xspextool_state

  cancel = 0

;  Check you are allowed to proceed
  
  if state.r.pscontinue lt 1 or state.r.xscontinue lt 1 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     cancel = 1
     return
     
  endif
  
;  Is there an actual wavecal file passed?
  
  if state.r.wavecal then begin
     
     awave = *state.d.awave
     atrans = *state.d.atrans
     
  endif

;  Construct the Profiles

  xspextool_message,'Constructing Spatial Profiles...'

  *state.r.profiles = mc_mkspatprof(*state.d.imgstruc,*state.r.orders,awave, $
                                    atrans,CANCEL=cancel)
  if cancel then return
  
;  Display the profiles

  if ~keyword_set(DOALLSTEPS) then begin
  
     doorders = intarr(state.r.norders)+1
     
     xmc_plotprofiles,*state.r.profiles,*state.r.orders,doorders,$
                      state.r.slith_arc,GROUP_LEADER=state.w.xspextool_base, $
                      POSITION=[0,0]

  endif
  
  state.r.pscontinue = 2
  state.r.xscontinue = 2
  
  xspextool_message,'Task complete.',/WINONLY

end
;
;******************************************************************************
;
pro xspextool_overridesigns

  common xspextool_state

  if state.r.pscontinue lt 4 then begin
     
     ok = dialog_message('Previous steps not complete.',/error,$
                         dialog_parent=state.w.xspextool_base)
     cancel = 1
     return
     
  endif
  
  string = mc_cfld(state.w.psapsigns_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  str = strsplit(string,',',/EXTRACT)
  
  apsigns = intarr(state.r.psnaps)
  z = where(str eq '+',count)
  if count ne 0 then apsigns[z] = 1
  z = where(str eq '-',count)
  if count ne 0 then apsigns[z] = -1
  
  *state.r.apsign = apsigns
  
  xspextool_message, 'Overriding Aperture signs: '+string
  
end
;
;******************************************************************************
;
pro xspextool_psapfind,DOALLSTEPS=doallsteps,CANCEL=cancel

  common xspextool_state

  cancel = 0
  
  if state.r.pscontinue lt 2 then begin
     
     ok = dialog_message('Previous steps not complete.',/error,$
                         dialog_parent=state.w.xspextool_base)
     cancel = 1
     return
     
  endif

  xspextool_message, 'Finding apertures positions...'

  if keyword_set(DOALLSTEPS) then begin

     doorders = *state.r.psdoorders
     appos = *state.r.appos
     guesspos = *state.r.guessappos

  endif else begin

     doorders = make_array(state.r.norders,/INTEGER,VALUE=1)
     guessnaps = state.r.psnaps
     fixnaps = state.r.psnaps

  endelse
     
  case state.r.psapfindmanmode of

     'Auto': begin
        
        xmc_plotprofiles,*state.r.profiles,*state.r.orders,doorders, $
                         state.r.slith_arc,AUTONAP=state.r.psnaps, $
                         POSITION=[0,0],GROUP_LEADER=state.w.xspextool_base, $
                         CANCEL=cancel
        
     end
     
     'Guess': begin
        
        xmc_plotprofiles,*state.r.profiles,*state.r.orders,doorders, $
                         state.r.slith_arc,GUESSPOS=guesspos, $
                         GUESSNAP=guessnaps,POSITION=[0,0], $
                         GROUP_LEADER=state.w.xspextool_base,$
                         CANCEL=cancel
        
     end
     
     'Fix': begin
        
        xmc_plotprofiles,*state.r.profiles,*state.r.orders,doorders, $
                         state.r.slith_arc,APPOS=appos,FIXNAP=fixnaps, $
                         POSITION=[0,0],GROUP_LEADER=state.w.xspextool_base,$
                         CANCEL=cancel
        
     end
     
  endcase
  
  state.r.pscontinue = 3

  xspextool_message,'Task complete.',/WINONLY
  
end
;
;******************************************************************************
;
pro xspextool_psdefineap,CANCEL=cancel

  common xspextool_state
  
  cancel = 0

;  Check you are allowed to proceed
  
  if state.r.pscontinue lt 4 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     cancel = 1
     return
     
  endif

;  Has the order selection changed?

  xmc_plotprofiles,GETINFO=plotprofileinfo

  if ~array_equal(*state.r.psdoorders,plotprofileinfo.doorders) then begin

     state.r.pscontinue = 3
     message = [['Number of orders to be extracted has changed.'],$
                ['Please trace the orders and define apertures again.']]
     junk = dialog_message(message,DIALOG_PARENT=state.w.xspextool_base, $
                                 /INFORMATION)
     cancel = 1
     return
    
  endif
    
  xspextool_message, 'Defining apertures...'

;  Get user inputs
  
  if state.r.optext[0] then begin
     
     psfradius = mc_cfld(state.w.psfradius_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     psfradius = mc_crange(psfradius,[0,state.r.slith_arc],'PSF Radius',/KGE,$
                           /KLE,WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then return
     
  endif
  
  apradius = mc_cfld(state.w.psapradius_fld,4,CANCEL=cancel,/EMPTY)
  if cancel then return
  
  range = (state.r.optext[0] eq 0) ? state.r.slith_arc:psfradius
  
  apradius = mc_crange(apradius,range,'Ap Radius',$
                    /KLE,WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
  if cancel then return
  
  sapradius = fltarr(state.r.psnaps,state.r.norders)

  if state.r.psbgsub eq 1 then begin

     bgstart = mc_cfld(state.w.psbgstart_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     bgstart = mc_crange(bgstart,apradius,'Background Start',/KGT,$
                         WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then return

     
     bgwidth = mc_cfld(state.w.psbgwidth_fld,4,CANCEL=cancel,/EMPTY)
     if cancel then return
     bgwidth = mc_crange(bgwidth,[0,state.r.slith_arc],'Background Width',/KGE,$
                      /KLE,WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then mc_setfocus,state.w.psbgwidth_fld
     if cancel then return
     
     psbginfo = [bgstart,bgwidth]
     
  endif else delvarx,psbginfo

  for i = 0, state.r.norders-1 do begin
     
     profile = (*state.r.profiles).(i)
     if (*state.r.psdoorders)[i] eq 1 then begin

;  Check to see the apertures don't fall off the slit
        
        tmp = [(*state.r.appos)[*,i]+apradius,(*state.r.appos)[*,i]-apradius]
        z = where(tmp lt 0 or tmp gt state.r.slith_arc,cnt)
        if cnt ne 0 then begin
           
           message=[['Some of the extraction apertures extend beyond ' + $
                     'the slit.'],['Please lower the aperture radius.']]
           junk = dialog_message(message,DIALOG_PARENT=state.w.xspextool_base, $
                                 /INFORMATION)
           cancel = 1
           return
           
        endif

;  create the slit masks
        
        m  = mc_mkapmask(profile[*,0],(*state.r.appos)[*,i],$
                         replicate(apradius,state.r.psnaps), $
                         PSBGINFO=psbginfo,WIDGET_ID=state.w.xspextool_base, $
                         CANCEL=cancel)
        if cancel then return
        
        word = 'mask'+strtrim(i+1,2)
        mask = (i eq 0) ? create_struct(word,[[reform(profile[*,0])],[m]]):$
               create_struct(mask,word,[[reform(profile[*,0])],[m]]) 
        
     endif else begin
        
        word = 'mask'+strtrim(i+1,2)
        b    = fltarr(n_elements(profile[*,1]))
        mask = (i eq 0) ? create_struct(word,[[reform(profile[*,0])],[b]]):$
               create_struct(mask,word,[[reform(profile[*,0])],[b]]) 
        
     endelse
     sapradius[*,i] = replicate(apradius,state.r.psnaps)
     
  endfor

;  Plot the apertures on ximgtool
  
  bot_tracecoeffs  = *state.r.tracecoeffs
  bot_tracecoeffs[0,*]  = bot_tracecoeffs[0,*]-total(apradius)

  struc = mc_tracetoxy(*state.d.omask,*state.d.wavecal,*state.d.spatcal, $
                       bot_tracecoeffs,state.r.psnaps,*state.r.orders, $
                       *state.r.psdoorders,CANCEL=cancel)
  if cancel then return
  
  bot = obj_new('mcplotap')
  bot->set,struc,3
    
  top_tracecoeffs = *state.r.tracecoeffs
  top_tracecoeffs[0,*] = top_tracecoeffs[0,*]+total(apradius)
  
  struc = mc_tracetoxy(*state.d.omask,*state.d.wavecal,*state.d.spatcal, $
                       top_tracecoeffs,state.r.psnaps,*state.r.orders, $
                       *state.r.psdoorders,CANCEL=cancel)
  if cancel then return
  
  top= obj_new('mcplotap')
  top->set,struc,3

  oplot = [bot,top]

  xspextool_ximgtool,OPLOT=oplot,CANCEL=cancel
  
;  Plot the apertures in xmc_plotprofiles  
  
  xmc_plotprofiles,*state.r.profiles,*state.r.orders,*state.r.psdoorders,$
                   state.r.slith_arc,APPOS=*state.r.appos, $
                   APRADII=replicate(apradius,state.r.psnaps), $
                   MASK=mask,GROUP_LEADER=state.w.xspextool_base, $
                   PSFAP=psfradius,POSITION=[0,0]
       
  state.r.pscontinue=5

  xspextool_message,'Task complete.',/WINONLY

end
;
;==============================================================================
;
pro xspextool_psextractspec,CANCEL=cancel

  common xspextool_state
  
  cancel = 0

;  Check you are allowed to proceed
  
  if state.r.pscontinue lt 5 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     cancel = 1
     return
     
  endif

;  Has the order selection changed?

  xmc_plotprofiles,GETINFO=plotprofileinfo

  if ~array_equal(*state.r.psdoorders,plotprofileinfo.doorders) then begin

     state.r.pscontinue = 3
     message = [['Number of orders to be extracted has changed.'],$
                ['Please trace the orders and define apertures again.']]
     junk = dialog_message(message,DIALOG_PARENT=state.w.xspextool_base, $
                                 /INFORMATION)
     cancel = 1
     return
    
  endif
  
  xspextool_message, 'Extracting spectra...'
  
;  If filename mode, check to make sure the outname is filled in.
  
  if state.r.filereadmode eq 'Filename' then begin
     
     file = mc_cfld(state.w.outname_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     
     junk = mc_fsextract(file,/FILENAME,CANCEL=cancel)
     if cancel then return
     
     if n_elements(junk) ne n_elements(*state.r.workfiles) then begin
        
        junk = dialog_message([['The number of output file names must'],$
                               [' equal the number of input images.']],$
                              DIALOG_PARENT=state.w.xspextool_base, $
                              /INFORMATION)
        cancel = 1
        return
        
     endif
     
  endif
  
;  Get user inputs

  if state.r.optext[0] then begin

     psfradius = mc_cfld(state.w.psfradius_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     psfradius = mc_crange(psfradius,[0,state.r.slith_arc],'PSF Radius',/KGE,$
                        /KLE,WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then return
     
  endif
      
  apradius = mc_cfld(state.w.psapradius_fld,4,CANCEL=cancel,/EMPTY)
  if cancel then return
  
  range = (state.r.optext[0] eq 0) ? state.r.slith_arc:psfradius
  
  apradius = mc_crange(apradius,range,'Ap Radius',/KLE, $
                    WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
  if cancel then return
  
  if state.r.psbgsub then begin
     
     bgstart = mc_cfld(state.w.psbgstart_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     bgstart = mc_crange(bgstart,apradius,'Background Start',/KGT,$
                      WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then return
     
     bgwidth = mc_cfld(state.w.psbgwidth_fld,4,CANCEL=cancel,/EMPTY)
     if cancel then return
     bgwidth = mc_crange(bgwidth,[0,state.r.slith_arc],'Background Width',$
                      /KGE,$
                      /KLE,WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then setfocus,state.w.psbgwidth_fld
     if cancel then return
     
     bgorder = mc_cfld(state.w.psbgfitdeg_fld,4,CANCEL=cancel,/EMPTY)
     if cancel then return
     bgorder = mc_crange(bgorder,[0,7],'Background Fit Order',/KGE,/KLE,$
                      WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then setfocus,state.w.psbgfitdeg_fld
     if cancel then return

     psbginfo = [bgstart,bgwidth,bgorder]
          
  endif

;  What orders are we extracting?

  zordr = where(*state.r.psdoorders eq 1,norders)
  
;  Create spatial map info if requested

  if state.r.fixbdpx[0] then begin

     thresh = mc_cfld(state.w.bdpxthresh_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     
     bpfinfo = {imgs:mc_indxstruc(*state.d.imgstruc,zordr), $
                profs:mc_indxstruc(*state.r.profiles,zordr), $
                medprof:state.r.medprofile[0],thresh:thresh}

     if state.r.wavecal then bpinfo = create_struct(bpfinfo,'awave', $
                                                    *state.d.awave, $
                                                    'atrans', $
                                                    *state.d.atrans)
     
  endif

  if state.r.optext[0] then begin

     if state.r.fixbdpx[0] then begin

        optinfo = create_struct(bpfinfo,'psfradius',psfradius)
        delvarx, bpfinfo
        
     endif else begin

        thresh = mc_cfld(state.w.bdpxthresh_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        
        optinfo = {imgs:mc_indxstruc(*state.d.imgstruc,zordr), $
                   profs:mc_indxstruc(*state.r.profiles,zordr), $
                   medprof:state.r.medprofile[0],thresh:thresh, $
                   psfradius:psfradius}

     endelse

     if state.r.wavecal then optinfo = create_struct(optinfo,'awave', $
                                                     *state.d.awave, $
                                                     'atrans',*state.d.atrans)

  endif

;  zordr = where(*state.r.orders eq 129)
;  orders = (*state.r.orders)[zordr]
;  img = *state.d.workimage
;  var = *state.d.varimage
;  bitmask = *state.d.bitmask
;  badmask = *state.d.bdpxmask
;  omask = *state.d.omask
;  wavecal = *state.d.wavecal
;  spatcal = *state.d.spatcal
;  edgecoeffs = (*state.r.edgecoeffs)[*,*,zordr]
;  tracecoeffs = *state.r.tracecoeffs
;  apsign = *state.r.apsign
;  imgstruc = mc_indxstruc(*state.d.imgstruc,zordr,CANCEL=cancel)
;  profstruc = mc_indxstruc(*state.r.profiles,zordr,CANCEL=cancel)
;     
;  save,FILENAME='save129.sav',img,var,bitmask,badmask,omask,orders,wavecal, $
;       spatcal,edgecoeffs,tracecoeffs,apradius,apsign,imgstruc,profstruc
;
;  return

;  s = mc_extpsspec(*state.d.workimage,*state.d.varimage, $
;                   intarr(state.r.ncols,state.r.nrows),*state.d.omask,$
;                   orders,*state.d.wavecal,*state.d.spatcal,$
;                   (*state.r.edgecoeffs)[*,*,z],*state.r.tracecoeffs,$
;                   apradius,*state.r.apsign,/UPDATE, $
;                   WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
;  if cancel then return


  s = mc_extpsspec(*state.d.workimage,*state.d.varimage,*state.d.bitmask,$
                   *state.d.omask,(*state.r.orders)[zordr],*state.d.wavecal,$
                   *state.d.spatcal,*state.r.tracecoeffs,apradius, $
                   *state.r.apsign,BPMASK=*state.d.bdpxmask, $
                   BGINFO=psbginfo,BPFINFO=bpfinfo,OPTINFO=optinfo,/UPDATE, $
                   WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
  if cancel then return
    
;  Convert output structure to old array format 

  ntags = n_tags(s)
  sizes = lonarr(2,ntags)
  for i = 0,ntags-1 do sizes[*,i] = size(s.(i),/DIMEN)

  spec = make_array(max(sizes),4,ntags,/DOUBLE,VALUE=!values.f_nan)
  for i = 0,ntags-1 do spec[0:(sizes[0,i]-1),*,i] = s.(i)
  
  *state.d.spectra = spec
  xspextool_writespec
  
  xspextool_message,'Task complete.',/WINONLY
  
end
;
;===============================================================================
;
pro xspextool_pstracespec, CANCEL=cancel

  common xspextool_state
  
  cancel = 0

;  Check you are allowed to proceed
  
  if state.r.pscontinue lt 3 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     cancel = 1
     return
     
  endif

;  Get info from xplotprofiles and save

  xmc_plotprofiles,GETINFO=plotprofileinfo,CANCEL=cancel
  if cancel then return

  *state.r.psdoorders = plotprofileinfo.doorders
  *state.r.guessappos = plotprofileinfo.guesspos
  *state.r.appos = plotprofileinfo.appos
  *state.r.apsign = plotprofileinfo.apsign

  string = strarr(state.r.psnaps)
  zz = where(*state.r.apsign eq -1,count)
  if count ne 0 then string[zz] = '-'
  zz = where(*state.r.apsign eq 1,count)
  if count ne 0 then string[zz] = '+'
  
  widget_control, state.w.psapsigns_fld[1],SET_VALUE=strjoin(string,',')
  xspextool_message, 'Aperture signs: '+strjoin(string,',')
  
;  Figure out what kind of aperture find we did

  case state.r.psapfindmanmode of

     'Auto': type = 0

     'Guess': type = 1

     'Fix': type = 2

  endcase
  if state.r.useappositions then type = 2

;  Now start the process of tracing the data
  
  case type of

     2: begin

;  Create a fake tracecoeffs array
     
        tracecoeffs = fltarr(2,total(*state.r.psdoorders)*state.r.psnaps)
     
        l = 0
        for i = 0,state.r.norders-1 do begin
           
           if ~(*state.r.psdoorders)[i] then continue
           
           for j = 0,state.r.psnaps-1 do begin
              
              tracecoeffs[*,l] = [(*state.r.appos)[j,i],0]
              l = l + 1
              
           endfor
           
        endfor
        *state.r.tracecoeffs = tracecoeffs

;  Generate objects to plot the results
        
        struc = mc_tracetoxy(*state.d.omask,*state.d.wavecal,*state.d.spatcal, $
                             tracecoeffs,state.r.psnaps,*state.r.orders, $
                             *state.r.psdoorders,CANCEL=cancel)
        if cancel then return
     
        oplot = obj_new('mcplotap')
        oplot->set,struc,7
              
     end

     else: begin

;  Get user inputs
     
        step = mc_cfld(state.w.tracestepsize_fld,3,CANCEL=cancel,/EMPTY)
        if cancel then return
        step = mc_crange(step,[1,20],'Step Size',/KGE,/KLE,$
                         WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then return
        
        sumap = mc_cfld(state.w.sumap_fld,3,CANCEL=cancel,/EMPTY)
        if cancel then return
        sumap = mc_crange(sumap,[1,2*sumap-1],'Colums Add',/KGE,/KLE,/ODD,$
                          WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then return
        
        fitdeg = mc_cfld(state.w.tracedeg_fld,3,CANCEL=cancel,/EMPTY)
        if cancel then return
        fitdeg = mc_crange(fitdeg,[0,4],'Trace Polynomial Degree',/KGE,/KLE,$
                           WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then return
        
        winthresh = mc_cfld(state.w.tracewinthresh_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        
        sigthresh = mc_cfld(state.w.tracesigthresh_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        
;  Trace objects
     
        xspextool_message, 'Tracing objects...'
        
        xspextool_ximgtool,WID=wid,PANNER=0,MAGNIFIER=0,CANCEL=cancel
        if cancel then return
        
        z = where(*state.r.psdoorders eq 1,norders)

        tracecoeffs = mc_tracespec(*state.d.workimage,*state.d.omask,$
                                   *state.d.wavecal,*state.d.spatcal, $
                                   (*state.r.orders)[z], $
                                   (*state.r.xranges)[*,z], $
                                   (*state.r.appos)[*,z], $
                                   step,sumap,winthresh,fitdeg,WID=wid, $
                                   OPLOT=oplot,CANCEL=cancel)
        if cancel then return
        *state.r.tracecoeffs = tracecoeffs

;  Generate objects to plot the results

        a = obj_new('mcoplot')
        a->set,oplot[*,0],oplot[*,1],PSYM=4,COLOR=3
        obj = a
        
        z = where(oplot[*,2] eq 0,cnt)
        if cnt ne 0 then begin
           
           b = obj_new('mcoplot')
           b = obj_new('mcoplot')
           b->set,oplot[z,0],oplot[z,1],PSYM=4,COLOR=4
           obj = [obj,b]
           
        endif
        
        struc = mc_tracetoxy(*state.d.omask,*state.d.wavecal,*state.d.spatcal, $
                             tracecoeffs,state.r.psnaps,*state.r.orders, $
                             *state.r.psdoorders,CANCEL=cancel)
        if cancel then return
        
;  Generate ximgtool data
        
        c = obj_new('mcplotap')
        c->set,struc,2
        
        oplot = [obj,c]
              
     end
   
  endcase
  
;  Display results on ximgtool

  xspextool_ximgtool,OPLOT=oplot,CANCEL=cancel
  if cancel then return

;  Write file to disk if requested

;  filename = mc_cfld(state.w.psotracefile_fld,7,CANCEL=cancel)
;  if cancel then return
;  
;  if filename ne '' then begin
;
;     z          = where(*state.r.psdoorders eq 1, gnorders)
;     goodorders = (*state.r.orders)[z]
;     naps       = state.r.psnaps
;     
;     fitorder = mc_cfld(state.w.tracedeg_fld,3,CANCEL=cancel,/EMPTY)
;     if cancel then return
;     fitorder = mc_crange(fitorder,[0,8],'Poly Fit Order',/KGE,/KLE,$
;                          WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
;     if cancel then return
;     
;     fullpath = state.r.calpath+filename+'_trace.dat'
;     
;     xspextool_message, [[['Writing trace information to']],[[fullpath]]]
;     
;     openw, lun, fullpath,/get_lun
;     
;     printf, lun, 'OBSMODE='+strtrim(state.r.obsmode,2)
;     printf, lun, 'DOORDERS='+strjoin(strtrim(*state.r.psdoorders,2),',')
;     printf, lun, 'ORDERS='+strjoin(strtrim(*state.r.orders,2),',')
;     printf, lun, 'NAPS='+strtrim(naps,2)
;     
;     for i = 0, state.r.norders-1 do begin
;        
;        printf, lun, 'ORDER'+string((*state.r.orders)[i],format='(i3.3)')+$
;                '_APPOS= '+strjoin(strtrim( (*state.r.appos)[*,i],2),', ')
;        
;     endfor
;     
;     for i = 0, gnorders-1 do begin
;        
;        idx = i*naps
;        for j = 0, naps-1 do begin
;           
;           printf, lun, 'ORDER'+string(goodorders[i],format='(i2.2)')+$
;                   '_AP'+string(j+1,format='(i2.2)')+'COEFF= '+$
;                   strjoin( strtrim( (*state.r.tracecoeffs)[*,idx],2),'  ' )
;           idx = idx + 1
;           
;        endfor
;        
;     endfor
;     
;     close, lun
;     free_lun, lun
;     xspextool_message,'Task complete.',/WINONLY
;     
;  endif
  
  state.r.pscontinue=4
  
  xspextool_message,'Task complete.',/WINONLY

end
;
;===============================================================================
;
pro xspextool_saveimages

  common xspextool_state

  index    = (state.r.filereadmode eq 'Index') ? 1:0
  filename = (state.r.filereadmode eq 'Filename') ? 1:0

;  Get outpaths.
  
  if index then begin
     
     oprefix = mc_cfld(state.w.oprefix_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     
  endif

  if filename then begin
     
     ofile = mc_cfld(state.w.outname_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return

  endif
  
;  Get user inputs.
  
  files = mc_cfld(state.w.sourceimages_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  files = mc_fsextract(files,INDEX=index,FILENAME=filename,CANCEL=cancel)
  if cancel then return

;  If A-B, must be even number of images
  
  if state.r.reductionmode eq 'A-B' then begin
     
     if n_elements(files) mod 2 ne 0 then begin
        
        mess = 'Must be an even number of images for pair subtraction.'
        ok = dialog_message(mess,/ERROR,DIALOG_PARENT=state.w.xspextool_base)
        mc_setfocus,state.w.sourceimages_fld
        cancel = 1
        return
        
     endif
     
  endif

;  If filename mode, must only be one image

  if filename then begin

     if n_elements(files) gt 1 then begin

        mess = 'Must be a single image.'
        ok = dialog_message(mess,/ERROR,DIALOG_PARENT=state.w.xspextool_base)
        mc_setfocus,state.w.sourceimages_fld
        cancel = 1
        return        

     endif

  endif
  
;  Get orders, flat name and arc name.
  
  orders = *state.r.orders
  xranges = *state.r.axranges
  
  flat = mc_cfld(state.w.flatfield_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  if state.r.wavecal then begin
     
     wavecal = mc_cfld(state.w.wavecal_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     wavetype = 'Vacuum'
     waveinfo = {wavecal:wavecal,wavetype:wavetype,wctype:state.r.wctype,$
                 disp:*state.d.disp}

     if strtrim(state.r.wctype,2) eq '2D' or $
        strtrim(state.r.wctype,2) eq '2DXD' then begin
        
        meth = (state.r.rectmethodmode[0] eq 0) ? 'interpolation':'polyclip'
        
        waveinfo = create_struct(waveinfo,'RECTMETH',meth)

     endif 
          
  endif 

  xunits= (state.r.wavecal eq 1) ? 'um':'pixels'
  yunits='DN / s'
  if state.r.lc[0] then linecormax = state.r.lincormax
  
;  Get loop size
  
  loopsize = (state.r.reductionmode eq 'A-B') ? (n_elements(files)/2.):$
             (n_elements(files))
 
  for i = 0, loopsize-1 do begin
     
     subset = (state.r.reductionmode eq 'A-B') ? files[i*2:i*2+1]:files[i]

     *state.r.workfiles = subset
     
;  Get output name

     if index then opath = state.r.procpath+oprefix+ $
                           strjoin(strtrim(*state.r.workfiles,2),'-')+'.fits'
     if filename then opath = state.r.procpath+ofile+'.fits'

;  Write the results out

     xspextool_loadimages,JUSTIMGS=i,CANCEL=cancel    
     if cancel then break


     if state.r.reductionmode eq 'A' then begin

        sky     = 'None'
        hdrinfo = (*state.r.hdrinfo)[0]
        aimage  = hdrinfo.vals.filename
        
     endif
     
     if state.r.reductionmode eq 'A-B' then begin

        aimage  = (*state.r.hdrinfo)[0].vals.filename                
        sky     = (*state.r.hdrinfo)[1].vals.filename
        hdrinfo = mc_avehdrs(*state.r.hdrinfo,/PAIR,CANCEL=cancel)
        if cancel then return
        
     endif
     
     if state.r.reductionmode eq 'A-Sky/Dark' then begin
        
        sky     = mc_cfld(state.w.skyimage_fld,7,/EMPTY,CANCEL=cancel)
        hdrinfo = (*state.r.hdrinfo)[0] 
        aimage  = hdrinfo.vals.filename

     endif

     xspextool_message,'Writing image to '+opath+'.'
     ximgtool,*state.d.bdpxmask
     
     mc_writeimg,opath,*state.d.workimage,*state.d.varimage,*state.d.bdpxmask,$
                 *state.d.bitmask,*state.d.wavecal,*state.d.spatcal, $
                 *state.r.xranges,aimage,sky,flat,*state.r.orders,hdrinfo, $
                 state.r.ps,state.r.slith_pix,state.r.slith_arc, $
                 state.r.slitw_pix,state.r.slitw_arc,state.r.rp,xunits, $
                 yunits,[state.r.ampcor[0],state.r.lc[0],state.r.flatfield[0]],$
                 state.w.version
         
  endfor

  xspextool_message,'Save Image(s) Complete.'
  






end
;
;===============================================================================
;
pro xspextool_updateappos,event

  common xspextool_state
  
  case state.w.base of 
     
     'Point Source': widget_control, state.w.psappos_fld[1],SET_VALUE=''
     
     'Extended Source': widget_control, state.w.xsappos_fld[1],SET_VALUE=''
     
  endcase
  
  state.r.pscontinue = 2
  state.r.xscontinue = 2
  
end
;
;==============================================================================
;
pro xspextool_useappositions,CANCEL=cancel

  common xspextool_state
  
  cancel = 0
  

  if state.w.base eq 'Point Source' then begin

     if state.r.pscontinue lt 3 then begin
        
        ok = dialog_message('Previous steps not complete.',/ERROR,$
                            DIALOG_PARENT=state.w.xspextool_base)
        cancel = 1
        return
        
     endif

     l = 0
     tracecoeffs = fltarr(2,total(*state.r.psdoorders)*state.r.psnaps)

     for i = 0,state.r.norders-1 do begin

        if (*state.r.psdoorders)[i] eq 0 then continue

        for j = 0,state.r.psnaps-1 do $
           tracecoeffs[*,l] = [(*state.r.appos)[j,i],0]


     endfor

;  Plot on ximgtool

     xx = rebin(indgen(state.r.ncols),state.r.ncols,state.r.nrows)
     yy = rebin(reform(indgen(state.r.nrows),1,state.r.nrows), $
                state.r.ncols,state.r.nrows)
     
     tri = (*state.d.tri)
     
     l = 0
     
     for i = 0,state.r.norders-1 do begin
        
        if (*state.r.psdoorders)[i] ne 1 then continue
        
        z = where(*state.d.omask eq (*state.r.orders)[i])
        
        wmax = max((*state.d.wavecal)[z],MIN=wmin)
        smax = max((*state.d.spatcal)[z],MIN=smin)
        xmax = max(xx[z],MIN=xmin)
        
        dw = (wmax-wmin)/(xmax-xmin)
        wave = (findgen(xmax-xmin-1)+1)*dw+wmin
        
        for j = 0,state.r.psnaps-1 do begin
           
           spat = poly(wave,coeffs[*,l])
           
           ix = trigrid((*state.d.wavecal)[z],(*state.d.spatcal)[z], $
                        xx[z],tri.(i),XOUT=wave,YOUT=spat, $
                        MISSING=!values.f_nan)
           
           iy = trigrid((*state.d.wavecal)[z],(*state.d.spatcal)[z], $
                        yy[z],tri.(i),XOUT=wave,YOUT=spat, $
                        MISSING=!values.f_nan)
           
           array = [[diag_matrix(ix)],[diag_matrix(iy)]]
           name = 'AP'+string(l+1,FORMAT='(I2.2)')
           
           struc = (l eq 0) ? $
                   create_struct(name,array):create_struct(struc,name,array)
           
           l = l + 1     
           
        endfor
        
     endfor
     
     a = obj_new('mcplotap')
     a->set,struc,7
     
     ximgtool,*state.d.workimage, WID=wid,GROUP_LEADER=state.w.xspextool_base,$
              /NOUPDATE,BUFFER=1,STDIMAGE=state.r.stdimage, $
              PLOTWINSIZE=state.r.plotwinsize,ZUNITS='DN / s',OPLOT=a,$
              POSITION=[1,0]
     
     state.r.pscontinue=4
     
     xspextool_message,'Task complete.',/WINONLY

  endif

  if state.w.base eq 'Extended Source' then begin


     if state.r.xscontinue lt 3 then begin
        
        ok = dialog_message('Previous steps not complete.',/ERROR,$
                            DIALOG_PARENT=state.w.xspextool_base)
        cancel = 1
        return
        
     endif
     
  endif

end
;
;******************************************************************************
;
pro xspextool_writespec,CANCEL=cancel

  common xspextool_state

  cancel = 0
  
  if state.w.base eq 'Point Source' then begin

     if state.r.optext[0] then begin
        
        psfradius = mc_cfld(state.w.psfradius_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        
     endif
          
     apradius = mc_cfld(state.w.psapradius_fld,4,CANCEL=cancel,/EMPTY)
     if cancel then return
     apradius = mc_crange(apradius,[0,state.r.slith_arc],'Aperture Radius', $
                          /KGE,/KLE,WIDGET_ID=state.w.xspextool_base, $
                          CANCEL=cancel)
     if cancel then mc_setfocus,state.w.psapradius_fld
     if cancel then return
     
     if state.r.psbgsub then begin
        
        bgstart = mc_cfld(state.w.psbgstart_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        bgstart = mc_crange(bgstart,apradius,'Background Start',/KGT,$
                         WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then return
        
        bgwidth = mc_cfld(state.w.psbgwidth_fld,4,CANCEL=cancel,/EMPTY)
        if cancel then return
        bgwidth = mc_crange(bgwidth,[0,state.r.slith_arc],'Background Width',$
                         /KGE,$
                         /KLE,WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then mc_setfocus,state.w.psbgwidth_fld
        if cancel then return
        
        bgorder = mc_cfld(state.w.psbgfitdeg_fld,4,CANCEL=cancel,/EMPTY)
        if cancel then return
        bgorder = mc_crange(bgorder,[0,7],'Background Fit Order',/KGE,/KLE,$
                         WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then mc_setfocus,state.w.psbgfitdeg_fld
        if cancel then return

        psbginfo = [bgstart,bgwidth,bgorder]

     endif

     good = where(*state.r.psdoorders eq 1,norders)
              
     if state.r.psapfindmode eq 'Import Coeffs' then begin
        
        tracefile = mc_cfld(state.w.psitracefile_fld,7,CANCEL=cancel,/EMPTY)
        if cancel then return
        junk = mc_cfile(state.r.calpath+tracefile,$
                        WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then return
        
     endif
     naps = state.r.psnaps
     
  endif

  if state.w.base eq 'Extended Source' then begin
     
     appos = (*state.r.appos)[*,0]
     
     apradius = mc_cfld(state.w.xsapradius_fld,7,CANCEL=cancel,/EMPTY)
     if cancel then return
     apradius = float( strsplit(temporary(apradius),',',/extract) )
     if n_elements(apradius) ne state.r.xsnaps then begin
        
        cancel = 1
        mess = 'Number of Apradii must equal number of Apertures.'
        ok = dialog_message(mess,/ERROR,DIALOG_PARENT=state.w.xspextool_base)
        mc_setfocus,state.w.xsapradius_fld
        return
        
        
     endif

     apradius = mc_crange(apradius,[0,state.r.slith_arc],'Apradius',/KGE,/KLE,$
                       WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then mc_setfocus,state.w.xsapradius_fld
     if cancel then return
     
     if state.r.xsbgsub then begin
        
        bgr  = mc_cfld(state.w.xsbg_fld,7,CANCEL=cancel,/EMPTY)
        if cancel then return
        bgorder = mc_cfld(state.w.xsbgfitdeg_fld,3,CANCEL=cancel,/EMPTY)
        if cancel then return
        bgorder = mc_crange(bgorder,[0,7],'Background Fit Order',/KGE,/KLE,$
                       WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then mc_setfocus,state.w.xsbgfitdeg_fld
        if cancel then return

        xsbginfo = [bgr,strtrim(bgorder,2)]

     endif
     
     good = where(*state.r.xsdoorders eq 1,norders)
     
;     if state.r.xsapfindmode eq 'Import Coeffs' then begin
;        
;        tracefile = mc_cfld(state.w.xsitracefile_fld,7,CANCEL=cancel,/EMPTY)
;        if cancel then return
;        junk = mc_cfile(state.r.calpath+tracefile,$
;                        WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
;        if cancel then return
;        
;     endif
     naps = state.r.xsnaps
     
  endif

;  Get file read mode.
  
  index    = (state.r.filereadmode eq 'Index') ? 1:0
  filename = (state.r.filereadmode eq 'Filename') ? 1:0
  
;  Get outpaths.
  
  if index then begin
     
     oprefix = mc_cfld(state.w.oprefix_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     files = *state.r.workfiles
     opaths = mc_mkfullpath(state.r.procpath,files,INDEX=index,$
                            FILENAME=filename,NI=state.r.nint,PREFIX=oprefix,$
                            WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     
  endif
  if filename then begin
     
     file = mc_cfld(state.w.outname_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     file = mc_fsextract(file,/FILENAME,CANCEL=cancel)
     if cancel then return
     opaths = state.r.procpath+file
     
  endif
  
;  Get orders, flat name and arc name.
  
  orders  = (*state.r.orders)[good]
  appos = (*state.r.appos)[*,good]
  xranges = (*state.r.axranges)[*,good]
  
  flat = mc_cfld(state.w.flatfield_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  if state.r.wavecal then begin
     
     wavecal = mc_cfld(state.w.wavecal_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     wavetype = 'Vacuum'
     disp = (*state.d.disp)[good]
     waveinfo = {wavecal:wavecal,wavetype:wavetype,wctype:state.r.wctype,$
                 disp:disp}

     if strtrim(state.r.wctype,2) eq '2D' or $
        strtrim(state.r.wctype,2) eq '2DXD' then begin
        
        meth = (state.r.rectmethodmode[0] eq 0) ? 'interpolation':'polyclip'
        
        waveinfo = create_struct(waveinfo,'RECTMETH',meth)

     endif 
          
  endif 

  xunits= (state.r.wavecal eq 1) ? 'um':'pixels'
  xtitle=(state.r.wavecal eq 1) ? '!7k!5 (!7l!5m)':'!7k!5 (pixels)' 
  yunits='DN / s'
  ytitle='f (!5DN s!u-1!N)'

  if state.r.lc[0] then linecormax = state.r.lincormax

  steps = [state.r.ampcor[0],state.r.lc[0],state.r.flatfield[0], $
           state.r.fixbdpx[0],state.r.optext[0]]

  if state.r.optext[0] then steps[3] = 0
  
  if state.w.base eq 'Point Source' then begin

     if state.r.reductionmode eq 'A' then begin

        xspextool_message,[[['Writing spectra to file']],[[opaths[0]+'.fits']]]
        
        sky                   = 'None'
        hdrinfo               = (*state.r.hdrinfo)[0] 
        aimage                = hdrinfo.vals.filename
        hdrinfo.vals.filename = file_basename(opaths[0])+'.fits'
        
        mc_writespec,*state.d.spectra,xranges,opaths[0]+'.fits',aimage,sky, $
                     flat,naps,orders,hdrinfo,appos,apradius,state.r.ps, $
                     state.r.slith_pix,state.r.slith_arc,state.r.slitw_pix, $
                     state.r.slitw_arc,state.r.rp,xunits,yunits,xtitle,ytitle, $
                     steps,state.w.version,PSFRADIUS=psfradius, $
                     PSBGINFO=psbginfo,WAVEINFO=waveinfo, $
                     LINCORMAX=state.r.lincormax,CANCEL=cancel
        if cancel then return
                
        xvspec,opaths[0]+'.fits',GROUP_LEADER=state.w.xspextool_base, $
               POSITION=[1,0.5],/PLOTLINMAX,/PLOTOPTFAIL
        
     endif
     
     if state.r.reductionmode eq 'A-B' then begin
        
;  Do any positive apertures.       
        
        pos = where(*state.r.apsign eq 1,pos_naps)
        if pos_naps ne 0 then begin
           
           xspextool_message,[[['Writing spectra to file']], $
                              [[opaths[0]+'.fits']]]
           
           hdrinfo               = (*state.r.hdrinfo)[0] 
           aimage                = hdrinfo.vals.filename
           sky                   = (*state.r.hdrinfo)[1].vals.filename
           hdrinfo.vals.filename = file_basename(opaths[0])+'.fits'
           
           
           print, *state.r.apsign
           apsign  = replicas(*state.r.apsign,norders)
           z       = where(apsign eq 1)
           
           mc_writespec,(*state.d.spectra)[*,*,z],xranges,opaths[0]+'.fits', $
                        aimage,sky,flat,pos_naps,orders,hdrinfo,appos[pos,*], $
                        apradius,state.r.ps,state.r.slith_pix, $
                        state.r.slith_arc,state.r.slitw_pix, $
                        state.r.slitw_arc,state.r.rp,xunits,yunits,xtitle, $
                        ytitle,steps,state.w.version,PSFRADIUS=psfradius, $
                        PSBGINFO=psbginfo,WAVEINFO=waveinfo, $
                        LINCORMAX=state.r.lincormax,CANCEL=cancel
           if cancel then return
           
        endif
        
;  Do negative apertures.
        
        neg = where((*state.r.apsign) eq -1,neg_naps)
        
        if neg_naps ne 0 then begin
           
           xspextool_message,[[['Writing spectra to file']], $
                              [[opaths[1]+'.fits']]]
           
           hdrinfo               = (*state.r.hdrinfo)[1] 
           aimage                = hdrinfo.vals.filename
           sky                   = (*state.r.hdrinfo)[0].vals.filename
           hdrinfo.vals.filename = file_basename(opaths[1])+'.fits'
           
           
           apsign  = replicas(*state.r.apsign,norders)
           z       = where(apsign eq -1)
           
           mc_writespec,(*state.d.spectra)[*,*,z],xranges,opaths[1]+'.fits', $
                        aimage,sky,flat,neg_naps,orders,hdrinfo,appos[neg,*], $
                        apradius,state.r.ps,state.r.slith_pix, $
                        state.r.slith_arc,state.r.slitw_pix, $
                        state.r.slitw_arc,state.r.rp,xunits,yunits,xtitle, $
                        ytitle,steps,state.w.version,PSFRADIUS=psfradius, $
                        PSBGINFO=psbginfo,WAVEINFO=waveinfo, $
                        LINCORMAX=state.r.lincormax,CANCEL=cancel
           if cancel then return
           
        endif
               
        if pos_naps and ~neg_naps then begin
           
           xvspec,opaths[0]+'.fits', $
                  GROUP_LEADER=state.w.xspextool_base,POSITION=[1,0.5], $
                  /PLOTLINMAX,/PLOTOPTFAIL
           
        endif
        
        if ~pos_naps and neg_naps then begin
           
           xvspec,opaths[1]+'.fits', $
                  GROUP_LEADER=state.w.xspextool_base,POSITION=[1,0.5], $
                  /PLOTLINMAX,/PLOTOPTFAIL
           
        endif
        
        if pos_naps and neg_naps then begin
           
           xvspec,opaths[0]+'.fits',opaths[1]+'.fits', $
                  GROUP_LEADER=state.w.xspextool_base,POSITION=[1,0.5], $
                  /PLOTLINMAX,/PLOTOPTFAIL
           
        endif
        
        
     endif
     
     if state.r.reductionmode eq 'A-Sky/Dark' then begin
        
        sky                   = mc_cfld(state.w.skyimage_fld,7,/EMPTY, $
                                        CANCEL=cancel)
        hdrinfo               = (*state.r.hdrinfo)[0] 
        aimage                = hdrinfo.vals.filename
        hdrinfo.vals.filename = file_basename(opaths[0])+'.fits'
        
        
        xspextool_message,[[['Writing spectra to file']],[[opaths[0]+'.fits']]]
        
        mc_writespec,*state.d.spectra,xranges,opaths[0]+'.fits',aimage, $
                     sky,flat,naps,orders,hdrinfo,appos,apradius,state.r.ps,$
                     state.r.slith_pix,state.r.slith_arc,state.r.slitw_pix, $
                     state.r.slitw_arc,state.r.rp,xunits,yunits,xtitle,ytitle, $
                     steps,state.w.version,PSFRADIUS=psfradius, $
                     PSBGINFO=psbginfo,WAVEINFO=waveinfo, $
                     LINCORMAX=state.r.lincormax,CANCEL=cancel
        if cancel then return
        
        xvspec,opaths[0]+'.fits',GROUP_LEADER=state.w.xspextool_base, $
               POSITION=[1,0.5],/PLOTLINMAX,/PLOTOPTFAIL
        
     endif

  endif


    if state.w.base eq 'Extended Source' then begin

     if state.r.reductionmode eq 'A' then begin

        xspextool_message,[[['Writing spectra to file']],[[opaths[0]+'.fits']]]
        
        sky                   = 'None'
        hdrinfo               = (*state.r.hdrinfo)[0] 
        aimage                = hdrinfo.vals.filename
        hdrinfo.vals.filename = file_basename(opaths[0])+'.fits'
        
        mc_writespec,*state.d.spectra,xranges,opaths[0]+'.fits',aimage, $
                     sky,flat,naps,orders,hdrinfo,appos,apradius,state.r.ps, $
                     state.r.slith_pix,state.r.slith_arc,state.r.slitw_pix, $
                     state.r.slitw_arc,state.r.rp,xunits,yunits,xtitle, $
                     ytitle,steps,state.w.version,XSBGINFO=xsbginfo, $
                     WAVEINFO=waveinfo,LINCORMAX=state.r.lincormax,CANCEL=cancel
        if cancel then return
        
        xvspec,opaths[0]+'.fits',GROUP_LEADER=state.w.xspextool_base, $
               POSITION=[1,0.5],/PLOTLINMAX,/PLOTOPTFAIL
        
     endif
     
     if state.r.reductionmode eq 'A-B' then begin
        
        xspextool_message,[[['Writing spectra to file']],[[opaths[0]+'.fits']]]
        
        hdrinfo               = (*state.r.hdrinfo)[0] 
        aimage                = hdrinfo.vals.filename
        sky                   = (*state.r.hdrinfo)[1].vals.filename
        hdrinfo.vals.filename = file_basename(opaths[0])+'.fits'
                
        mc_writespec,(*state.d.spectra),xranges,opaths[0]+'.fits',aimage, $
                     sky,flat,naps,orders,hdrinfo,appos,apradius,state.r.ps, $
                     state.r.slith_pix,state.r.slith_arc,state.r.slitw_pix, $
                     state.r.slitw_arc,state.r.rp,xunits,yunits,xtitle,ytitle, $
                     steps,state.w.version,XSBGINFO=xsbginfo, $
                     WAVEINFO=waveinfo,LINCORMAX=state.r.lincormax,CANCEL=cancel
        if cancel then return
        
        xvspec,opaths[0]+'.fits', $
               GROUP_LEADER=state.w.xspextool_base,POSITION=[1,0.5], $
               /PLOTLINMAX,/PLOTOPTFAIL
        
     endif
     
     if state.r.reductionmode eq 'A-Sky/Dark' then begin
        
        sky = mc_cfld(state.w.skyimage_fld,7,/EMPTY,CANCEL=cancel)
        if cancel then return

        hdrinfo               = (*state.r.hdrinfo)[0] 
        aimage                = hdrinfo.vals.filename
        hdrinfo.vals.filename = file_basename(opaths[0])+'.fits'
        
        
        xspextool_message,[[['Writing spectra to file']],[[opaths[0]+'.fits']]]
        
        mc_writespec,*state.d.spectra,xranges,opaths[0]+'.fits',aimage, $
                     sky,flat,naps,orders,hdrinfo,appos,apradius,state.r.ps, $
                     state.r.slith_pix,state.r.slith_arc,state.r.slitw_pix, $
                     state.r.slitw_arc,state.r.rp,xunits,yunits,xtitle,ytitle, $
                     steps,state.w.version,XSBGINFO=xsbginfo, $
                     WAVEINFO=waveinfo,LINCORMAX=state.r.lincormax,CANCEL=cancel
        if cancel then return
        
        xvspec,opaths[0]+'.fits',GROUP_LEADER=state.w.xspextool_base, $
               POSITION=[1,0.5],/PLOTLINMAX,/PLOTOPTFAIL
        
     endif

  endif
 
end
;
;******************************************************************************
;
pro xspextool_ximgtool,WID=wid,OPLOT=oplot,PANNER=panner,MAGNIFIER=magnifier,$
                       FORCE=force,CANCEL=cancel

  common xspextool_state

  cancel = 0

  edges = obj_new('mcoplotedge')

  edges->set,*state.r.xranges,*state.r.edgecoeffs,ORDERS=*state.r.orders

  speccoords = {ordr:*state.d.omask,$
                wave:*state.d.wavecal,wunits:'um',wfmt:state.r.wavefmt, $
                spat:*state.d.spatcal,sunits:'"',sfmt:state.r.spatfmt}
  
  if keyword_set(OPLOT) then oplot = [oplot,edges] else oplot=[edges]
  
  if xregistered('ximgtool') and ~keyword_set(FORCE) then begin
    
     ximgtool,*state.d.workimage/sqrt(*state.d.varimage),/NOUPDATE, $
              BUFFER=3,OPLOT=oplot,/NODISPLAY

     ximgtool,sqrt(*state.d.varimage),BUFFER=2,OPLOT=oplot,/NOUPDATE,/NODISPLAY
     
     ximgtool,*state.d.workimage,BUFFER=1,OPLOT=oplot,/NOUPDATE,/LOCK, $
              WID=wid,PANNER=panner,MAGNIFIER=magnifier

  endif else begin
     
     ximgtool,*state.d.bdpxmask,$
              BUFFER=4,ZRANGE=[0,1],$
              WID=wid,GROUP_LEADER=state.w.xspextool_base, $
              STD=state.r.stdimage,PLOTWINSIZE=state.r.plotwinsize,$
              POSITION=[1,0],FILENAME='Bad Pixel Mask', $
              BITMASK={bitmask:*state.d.bitmask,plot1:[0,2]},$
              OPLOT=oplot,/NODISPLAY,/ZTOFIT
     
     ximgtool,*state.d.workimage/sqrt(*state.d.varimage),$
              BUFFER=3,RANGE='98%',$
              WID=wid,GROUP_LEADER=state.w.xspextool_base, $
              STD=state.r.stdimage,PLOTWINSIZE=state.r.plotwinsize,$
              POSITION=[1,0],FILENAME='S/N', $
              BITMASK={bitmask:*state.d.bitmask,plot1:[0,2]},$
              OPLOT=oplot,/NODISPLAY,/ZTOFIT
     
     ximgtool,sqrt(*state.d.varimage),$
              BUFFER=2,RANGE='98%',$
              WID=wid,GROUP_LEADER=state.w.xspextool_base, $
              STD=state.r.stdimage,PLOTWINSIZE=state.r.plotwinsize,$
              ZUNITS='(DN / s)',POSITION=[1,0],FILENAME='Uncertainty', $
              BITMASK={bitmask:*state.d.bitmask,plot1:[0,2]},$
              OPLOT=oplot,/NODISPLAY,/ZTOFIT
     
     ximgtool,*state.d.workimage,$
              BUFFER=1,$
              RANGE='98%',$
              WID=wid,GROUP_LEADER=state.w.xspextool_base, $
              STD=state.r.stdimage,PLOTWINSIZE=state.r.plotwinsize,$
              ZUNITS='(DN / s)',POSITION=[1,0],FILENAME='Flux',/LOCK,$
              BITMASK={bitmask:*state.d.bitmask,plot1:[0,2]},OPLOT=oplot, $
              ZTOFIT=1,SPECCOORDS=speccoords,PANNER=panner,MAGNIFIER=magnifier
;     ,$
;              XYCENTER=[982,1680],ZOOM=3,ZRANGE=[-0.1,5.5]
     
  endelse
  
end
;
;******************************************************************************
;
pro xspextool_xsapfind,DOALLSTEPS=doallsteps,CANCEL=cancel

  common xspextool_state

  cancel = 0
  
  if state.r.xscontinue lt 2 then begin
     
     ok = dialog_message('Previous steps not complete.',/error,$
                         dialog_parent=state.w.xspextool_base)
     cancel = 1
     return
     
  endif
  
  xspextool_message, 'Finding aperture positions...'

  if ~keyword_set(DOALLSTEPS) then begin
  
     case  state.r.xsapfindmode of
     
        'Cursor': begin
           
           naps = mc_cfld(state.w.xsnaps_fld,3,/EMPTY,CANCEL=cancel)
           if cancel then return
           
           naps = mc_crange(naps,[0,5],'Number of Apertures',/KGT,/KLE,$
                            WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
           if cancel then begin
              
              mc_setfocus, state.w.xsnaps_fld
              return
              
           endif
           
           state.r.xsnaps = naps
           xmc_plotprofiles,*state.r.profiles,*state.r.orders, $
                            make_array(state.r.norders,/INTEGER,VALUE=1), $
                            state.r.slith_arc,POSITION=[0,0],FIXNAP=naps,$
                            GROUP_LEADER=state.w.xspextool_base
           
        end
        
        'Keyboard': begin
           
           pos = mc_cfld(state.w.xsappos_fld,7,/EMPTY,CANCEL=cancel)
           if cancel then return
           
           pos = float( strsplit(temporary(pos),',',/extract) )
           pos = mc_crange(pos,[0,state.r.slith_arc],'Pos',/KGE,/KLE,$
                           WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
           
           if cancel then begin
              
              mc_setfocus, state.w.xsappos_fld
              return    
              
           endif
           
           state.r.xsnaps = n_elements(pos)
           
           pos = rebin(pos,state.r.xsnaps,state.r.norders)
           xmc_plotprofiles,*state.r.profiles,*state.r.orders,$
                            intarr(state.r.norders)+1,state.r.slith_arc, $
                            APPOS=pos,GROUP_LEADER=state.w.xspextool_base, $
                            POSITION=[0,0]
           
        end
        
     endcase

  endif else begin


     xmc_plotprofiles,*state.r.profiles,*state.r.orders,*state.r.xsdoorders, $
                      state.r.slith_arc,APPOS=*state.r.appos, $
                      GROUP_LEADER=state.w.xspextool_base,POSITION=[0,0]
     
  endelse
     
  state.r.xscontinue = 3  
  xspextool_message,'Task complete.',/WINONLY

end
;
;******************************************************************************
;
pro xspextool_xsdefineap,CANCEL=cancel

  common xspextool_state

  cancel = 0

  if state.r.xscontinue lt 4 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     cancel = 1
     return
     
  endif

;  Has the order selection changed?

  xmc_plotprofiles,GETINFO=plotprofileinfo

  if ~array_equal(*state.r.xsdoorders,plotprofileinfo.doorders) then begin

     state.r.pscontinue = 3
     message = [['Number of orders to be extracted has changed.'],$
                ['Please trace the orders and define apertures again.']]
     junk = dialog_message(message,DIALOG_PARENT=state.w.xspextool_base, $
                                 /INFORMATION)
     cancel = 1
     return
    
  endif
  
  xspextool_message, 'Defining apertures...'

;  Get user inputs.

  apradius = mc_cfld(state.w.xsapradius_fld,7,CANCEL=cancel,/EMPTY)
  if cancel then return
  apradius = float( strsplit(temporary(apradius),',',/EXTRACT) )

;  Check number and range of aperture radii

  if n_elements(apradius) ne state.r.xsnaps then begin
     
     cancel = 1
     mess = 'Number of Apradii must equal number of Apertures.'
     ok = dialog_message(mess,/error,dialog_parent=state.w.xspextool_base)
     mc_setfocus,state.w.xsapradius_fld
     return
     
     
  endif
  apradius = mc_crange(apradius,[0,state.r.slith_arc],'Apradius',/KGE,/KLE,$
                    WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
  if cancel then mc_setfocus,state.w.xsapradius_fld
  if cancel then return

;  Get background regions
  
  if state.r.xsbgsub then begin
     
     bg = mc_cfld(state.w.xsbg_fld,7,CANCEL=cancel,/EMPTY)
     if cancel then return
     bg = strsplit(temporary(bg),',',/extract) 
     for i = 0, n_elements(bg)-1 do begin
        
        junk = float( strsplit(temporary(bg[i]),'-',/extract) )
        bgr = (i eq 0) ? junk:[[bgr],[junk]]
        
     endfor
     
     bgfit = mc_cfld(state.w.xsbgfitdeg_fld,3,CANCEL=cancel,/EMPTY)
     if cancel then return
     bgfit = mc_crange(bgfit,[0,7],'Background Fit Order',/KGE,/KLE,$
                    WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then setfocus,state.w.xsbgfitdeg_fld
     if cancel then return
     
  endif

;  Get plotting stuff

  sapradius = fltarr(state.r.xsnaps,state.r.norders)
  for i = 0, state.r.norders-1 do begin
     
     profile = (*state.r.profiles).(i)
     if (*state.r.xsdoorders)[i] eq 1 then begin
        
        tmp = [(*state.r.appos)[*,i]+apradius,(*state.r.appos)[*,i]-apradius]
        z = where(tmp lt 0 or tmp gt state.r.slith_arc,cnt)
        if cnt ne 0 then begin
           
        message=[['Some of the extraction apertures extend beyong the slit.'],$
                 ['Please lower the aperture radius.']]
           junk = dialog_message(message,DIALOG_PARENT=state.w.xspextool_base, $
                                 /INFORMATION)
           return
           
        endif
        
        m  = mc_mkapmask(profile[*,0],(*state.r.appos)[*,i],apradius, $
                         XSBGINFO=bgr,WIDGET_ID=state.w.xspextool_base,$
                         CANCEL=cancel)
        if cancel then return

        word = 'mask'+strtrim(i+1,2)
        mask = (i eq 0) ? create_struct(word,[[reform(profile[*,0])],[m]]):$
               create_struct(mask,word,[[reform(profile[*,0])],[m]]) 
        
     endif else begin
        
        word = 'mask'+strtrim(i+1,2)
        b    = fltarr(n_elements(profile[*,1]))
        mask = (i eq 0) ? create_struct(word,[[reform(profile[*,0])],[b]]):$
               create_struct(mask,word,[[reform(profile[*,0])],[b]]) 
        
     endelse
     sapradius[*,i] = apradius
     
  endfor

; Plot on ximgtool.  Must do for each aperture since they can have
; different apradii

  junk = where(*state.r.xsdoorders eq 1,cnt)
  

  low_tracecoeffs  = *state.r.tracecoeffs
  high_tracecoeffs = *state.r.tracecoeffs

  for i = 0,cnt-1 do begin     

     for j = 0,state.r.xsnaps-1 do begin

        low_tracecoeffs[0,i+j*(state.r.xsnaps-1)]  = $
           low_tracecoeffs[0,i+j*(state.r.xsnaps-1)] - apradius[j]

     endfor

  endfor

  struc = mc_tracetoxy(*state.d.omask,*state.d.wavecal,*state.d.spatcal, $
                       low_tracecoeffs,state.r.xsnaps,*state.r.orders, $
                       *state.r.xsdoorders,CANCEL=cancel)
  if cancel then return

  c = obj_new('mcplotap')
  c->set,struc,3

  for i = 0,cnt-1 do begin

     for j = 0,state.r.xsnaps-1 do begin

        high_tracecoeffs[0,i+j*(state.r.xsnaps-1)]  = $
           high_tracecoeffs[0,i+j*(state.r.xsnaps-1)] + apradius[j]

     endfor

  endfor
  
  struc = mc_tracetoxy(*state.d.omask,*state.d.wavecal,*state.d.spatcal, $
                       high_tracecoeffs,state.r.xsnaps,*state.r.orders, $
                       *state.r.xsdoorders,CANCEL=cancel)
  if cancel then return
  
  d = obj_new('mcplotap')
  d->set,struc,3

  oplot = [c,d]

  xspextool_ximgtool,OPLOT=oplot,CANCEL=cancel
  if cancel then return
  
  xmc_plotprofiles,*state.r.profiles,*state.r.orders,*state.r.xsdoorders,$
                   state.r.slith_arc,APPOS=*state.r.appos,APRADII=apradius, $
                   MASK=mask,GROUP_LEADER=state.w.xspextool_base,POSITION=[0,0]
  
  state.r.xscontinue = 5


  
  xspextool_message,'Task complete.',/WINONLY
  
end
;
;==============================================================================
;
pro xspextool_xsextractspec,CANCEL=cancel

  common xspextool_state
  
  cancel = 0
    
  if state.r.xscontinue lt 5 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     cancel = 1
     return
     
  endif

;  Has the order selection changed?

  xmc_plotprofiles,GETINFO=plotprofileinfo

  if ~array_equal(*state.r.xsdoorders,plotprofileinfo.doorders) then begin

     state.r.pscontinue = 3
     message = [['Number of orders to be extracted has changed.'],$
                ['Please trace the orders and define apertures again.']]
     junk = dialog_message(message,DIALOG_PARENT=state.w.xspextool_base, $
                                 /INFORMATION)
     cancel = 1
     return
    
  endif
  
  xspextool_message, 'Extracting spectra...'
  
;  If filename mode, check to make sure the outname is filled in.
  
  if state.r.filereadmode eq 'Filename' then begin
     
     file = mc_cfld(state.w.outname_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     
     junk = mc_fsextract(file,/FILENAME,CANCEL=cancel)
     if cancel then return
     
     if n_elements(junk) ne n_elements(*state.r.workfiles) then begin
        
        junk = dialog_message([['The number of output file names must'],$
                               [' equal the number of input images.']],$
                              DIALOG_PARENT=state.w.xspextool_base, $
                              /INFORMATION)
        cancel = 1
        return
        
     endif
     
  endif
  
;  Get user inputs
  
  apradii = mc_cfld(state.w.xsapradius_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  apradii = float( strsplit(temporary(apradii),',',/EXTRACT) )
  if n_elements(apradii) ne state.r.xsnaps then begin
     
     cancel = 1
     ok = dialog_message('Number of Apradii must equal ' + $
                         'number of Apertures.',$
                         /ERROR,DIALOG_PARENT=state.w.xspextool_base)
     mc_setfocus,state.w.xsapradius_fld
     return
     
     
  endif
  
  apradii = mc_crange(apradii,[0,state.r.slith_arc],'Apradius',/KGE,/KLE,$
                      WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
  if cancel then mc_setfocus,state.w.xsapradius_fld
  if cancel then return
  
  if state.r.xsbgsub then begin
     
     bg  = mc_cfld(state.w.xsbg_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     bg = strsplit(temporary(bg),',',/EXTRACT) 
     for i = 0, n_elements(bg)-1 do begin
        
        junk = float( strsplit(temporary(bg[i]),'-',/EXTRACT) )
        bgr = (i eq 0) ? junk:[[bgr],[junk]]
        
     endfor
     
     bgdeg = mc_cfld(state.w.xsbgfitdeg_fld,3,/EMPTY,CANCEL=cancel)
     if cancel then return
     bgdeg = mc_crange(bgdeg,[0,7],'Background Fit Order',/KGE,/KLE,$
                       WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then setfocus,state.w.xsbgfitdeg_fld
     if cancel then return
     
     xsbginfo = {bgreg:bgr,bgdeg:bgdeg}
     
  endif
  
  zordr = where(*state.r.xsdoorders eq 1,norders)
  
;  Create spatial map info if requested
  
  if state.r.fixbdpx[0] then begin
     
     thresh = mc_cfld(state.w.bdpxthresh_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     
     bpfinfo = {imgs:mc_indxstruc(*state.d.imgstruc,zordr), $
                profs:mc_indxstruc(*state.r.profiles,zordr), $
                medprof:state.r.medprofile[0],thresh:thresh}
     
     if state.r.wavecal then $
        bpinfo = create_struct(bpfinfo,'awave',*state.d.awave, $
                               'atrans',*state.d.atrans)
     
  endif
  
  s = mc_extxsspec(*state.d.workimage,*state.d.varimage,*state.d.bitmask,$
                   *state.d.omask,(*state.r.orders)[zordr],*state.d.wavecal,$
                   *state.d.spatcal,*state.r.tracecoeffs,apradii, $
                   BPMASK=*state.d.bdpxmask,BGINFO=xsbginfo,BPFINFO=bpfinfo, $
                   /UPDATE,WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
  if cancel then return
   
;  if state.r.wctype eq '2D' then begin
;
;     s = mc_extspec2d2(*state.d.workimage,*state.d.varimage,*state.d.omask, $
;                       orders,*state.d.wavecal,*state.d.spatcal, $
;                       *state.r.edgecoeffs,state.r.ps,*state.r.apsign, $
;                       *state.r.tracecoeffs,apradii, $
;                       PSBGINFO=xsbginfo,BGDEG=bgorder, $
;                       SMAPINFO=smapinfo,/UPDATE, $
;                       WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
;     if cancel then return
;     
;;     s = mc_extspec2d(*state.d.workimage,*state.d.varimage,*state.d.omask, $
;;                      orders,*state.d.wavecal,*state.d.spatcal,state.r.ps, $
;;                      *state.r.apsign,*state.r.tracecoeffs,apradii, $
;;                      XSBGINFO=bgr,BGDEG=bgorder,SMAPINFO=smapinfo, $
;;                      /UPDATE,WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
;;     if cancel then return
;         
;  endif
  
;  Convert output structure to old array format and compute
;  approximate dispersion.

  ntags = n_tags(s)
  sizes = lonarr(2,ntags)
    for i = 0,ntags-1 do sizes[*,i] = size(s.(i),/DIMEN)
  
  spec = dblarr(max(sizes),4,ntags)*!values.f_nan
  for i = 0,ntags-1 do spec[0:(sizes[0,i]-1),*,i] = s.(i)
  
  *state.d.spectra = spec
  xspextool_writespec
  
  xspextool_message,'Task complete.',/WINONLY

end
;
;==============================================================================
;
pro xspextool_xstracespec,CANCEL=cancel

  cancel = 0
  
  common xspextool_state
  
  cancel = 0
   
  if state.r.xscontinue lt 3 then begin
     
     ok = dialog_message('Previous steps not complete.',/ERROR,$
                         DIALOG_PARENT=state.w.xspextool_base)
     cancel = 1
     return
     
  endif

;  Get info from xplotprofiles

  xmc_plotprofiles,GETINFO=info,CANCEL=cancel
  if cancel then return
  *state.r.xsdoorders = info.doorders
  *state.r.appos = info.appos
  *state.r.guessappos = info.guesspos
  *state.r.apsign = replicate(1,state.r.xsnaps)
  
;  Create fake trace

  coeffs = fltarr(2,total(*state.r.xsdoorders)*state.r.xsnaps)

  l = 0
  for i = 0,state.r.norders-1 do begin
     
     if (*state.r.xsdoorders)[i] eq 0 then continue

     for j = 0,state.r.xsnaps-1 do begin

        coeffs[*,l] = [(info.appos)[j,i],0]
        l =l + 1
        
     endfor
     
  endfor
  *state.r.tracecoeffs = coeffs

;  Plot on ximgtool  

  struc = mc_tracetoxy(*state.d.omask,*state.d.wavecal,*state.d.spatcal, $
                       coeffs,state.r.xsnaps,*state.r.orders, $
                       *state.r.xsdoorders,CANCEL=cancel)
  if cancel then return     

     
;  Plot on ximgtool
  
  oplot = obj_new('mcplotap')
  oplot->set,struc,7

  xspextool_ximgtool,OPLOT=oplot,CANCEL=cancel
  
  state.r.xscontinue=4
  
  xspextool_message,'Task complete.',/WINONLY

end
;
;******************************************************************************
;
pro xspextool,instrument,FAP=fap,ENG=eng,IPREFIX=iprefix,CANCEL=cancel

  cancel = 0

;  See if you have one running.

  if xregistered('xspextool') then begin

     print
     print, 'Spextool is already running.'
     print
     cancel = 1
     return

  endif

  mc_mkct
  
;  Start it up
  
  xspextool_startup,INSTRUMENT=instrument,FAB=fap,ENG=eng,IPREFIX=iprefix, $
                    CANCEL=cancel
  if cancel then return

end
     
