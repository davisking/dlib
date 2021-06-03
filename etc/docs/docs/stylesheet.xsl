<?xml version="1.0" encoding="utf8" ?>
   
<!--
   To the extent possible under law, Davis E King  has waived all copyright and 
   related or neighboring rights to dlib documentation (XML, HTML, and XSLT files).
   This work is published from United States. 
-->

   <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
   xmlns:gcse="googleCustomSearch">
   <xsl:output method='html' version='1.0' encoding='UTF-8' indent='no' />
   <xsl:strip-space elements="*" />
   <xsl:preserve-space elements="pre code_box preserve_space" />
   
   
   <!-- ************************************************************************* -->
   
   <xsl:variable name="is_chm">true</xsl:variable>
   <xsl:variable name="main_menu">main_menu.xml</xsl:variable>
   <xsl:variable name="project_name">dlib C++ Library</xsl:variable>
   
   <!-- ************************************************************************* -->

   <!-- This is the ID you get from Google Webmaster Tools -->
   <xsl:variable name="google_verify_id">02MiiaFNVzS5/u0eQhsy3/knioFHsia1X3DXRpHkE6I=</xsl:variable>
   <xsl:variable name="google_verify_id2">DGSSJMKDomaDaDTIRJ8jDkv0YMx9Cz7OESbXHjjr6Jw</xsl:variable>

   <!-- ************************************************************************* -->

   <xsl:variable name="last_modified_date_var">_LAST_MODIFIED_DATE_</xsl:variable>
   <xsl:variable name="current_release_var">_CURRENT_RELEASE_</xsl:variable>
   <xsl:template match="last_modified_date"><xsl:value-of select="$last_modified_date_var"/></xsl:template>
   <xsl:template match="current_release"><xsl:value-of select="$current_release_var"/></xsl:template>
   
   <!-- ************************************************************************* -->
   
   <xsl:variable name="gray">#E3E3E3</xsl:variable>

   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->

   <xsl:variable name="lcletters">abcdefghijklmnopqrstuvwxyz </xsl:variable>
   <xsl:variable name="ucletters">ABCDEFGHIJKLMNOPQRSTUVWXYZ </xsl:variable>
   <xsl:variable name="badletters">'?()&lt;&gt; /\&amp;~!@#$%^*_+=-[]{}</xsl:variable>
   
   <!-- ************************************************************************* -->

   <xsl:template match="/doc">
      <html>
         <head>
            <link rel="shortcut icon" href="dlib-icon.ico"/>
            <xsl:if test="$is_chm != 'true'">
               <meta property="og:image" content="http://dlib.net/dlib-logo-small.png"/>
            </xsl:if>

            <!-- Verify with Google -->
            <meta name="verify-v1" content="{$google_verify_id}" />
            <meta name="google-site-verification" content="{$google_verify_id2}" />
            <title>
               <xsl:value-of select="$project_name"/>
               <xsl:if test="title">
               - <xsl:value-of select="title" />
               </xsl:if>
            </title>

            <script type="text/javascript" src="dlib.js"></script>
            <link rel="stylesheet" type="text/css" href="dlib.css"/>

            <xsl:if test="$is_chm != 'true'">
               <script> <!-- Google Analytics -->
               (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
               (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
               m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
               })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

               ga('create', 'UA-51919357-1', 'dlib.net');
               ga('send', 'pageview');
               </script>
            </xsl:if>
         </head>



         <body>
            <a name="top" />
            <div id="page_header">
               <xsl:if test="$is_chm != 'true'">
                  <div style="float:right;width:450px">
                           <script>
                              (function() {
                                 var cx = '017764538452368798135:fin3a18x_ns';
                                 var gcse = document.createElement('script');
                                 gcse.type = 'text/javascript';
                                 gcse.async = true;
                                 gcse.src = (document.location.protocol == 'https:' ? 'https:' : 'http:') +
                                    '//www.google.com/cse/cse.js?cx=' + cx;
                                 var s = document.getElementsByTagName('script')[0];
                                 s.parentNode.insertBefore(gcse, s);
                              })();
                        </script><gcse:search></gcse:search>
                  </div>
               </xsl:if>
               <a href="http://dlib.net"><img src="dlib-logo.png"/></a>
            </div>


            <div id="top_content">
               <div id="main_menu" class="menu">
                  <xsl:apply-templates select="document($main_menu)/doc/menu"/>
               </div>

               <!-- ************************************************************************* -->
               <div id="main_text">
                  <xsl:if test="title">
                     <div id="main_text_title"> <xsl:value-of select="title" /> </div>
                  </xsl:if>

                  <div id="main_text_body">
                     <xsl:apply-templates select="body"/>
                     <xsl:for-each select="questions">
                        <xsl:sort select="translate(@group,$lcletters, $ucletters)"/> 
                        <xsl:if test="@group"><h2><xsl:value-of select="@group"/></h2></xsl:if>
                        <ul>
                        <xsl:for-each select="question">      
                           <xsl:sort select="translate(@text,$lcletters, $ucletters)"/> 
                           <li><a href="#{translate(@text,$badletters,'')}"><xsl:value-of select="@text"/></a></li>
                        </xsl:for-each>
                        </ul>
                     </xsl:for-each>
                  </div>
               </div>

               <!-- ************************************************************************* -->
                  <xsl:choose>
                     <xsl:when test="menu/@from_file">
                        <div id="right_menu" class="menu">
                           <xsl:apply-templates select="document(menu/@from_file)/doc/menu">
                              <xsl:with-param name="file_name" select="concat(substring-before(menu/@from_file,'.'),'.html')" />
                           </xsl:apply-templates>
                        </div>
                     </xsl:when>
                     <xsl:otherwise>
                        <xsl:if test="menu">
                           <div id="right_menu" class="menu">
                              <xsl:apply-templates select="menu"/>
                           </div>
                        </xsl:if>
                     </xsl:otherwise>
                  </xsl:choose>         
               <!-- ************************************************************************* -->
            </div>
                  
            <div id="bottom_content">
               <xsl:apply-templates select="components"/>
               <xsl:apply-templates select="questions"/>
            </div>

         </body>
      </html>
   </xsl:template>
   
         


   
   <!-- ************************************************************************* -->

   <xsl:template match="section">
      <xsl:param name="file_name" />
      <b><xsl:value-of select="name"/></b>
      <ul  class="tree">
         <xsl:if test="$is_chm = 'true'">
            <xsl:for-each select="item | chm/item">  
            <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/> 
               <xsl:apply-templates select=".">
                  <xsl:with-param name="file_name" select="$file_name" />
               </xsl:apply-templates>
            </xsl:for-each>
         </xsl:if>
         <xsl:if test="$is_chm != 'true'">
            <xsl:for-each select="item | web/item">  
            <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/> 
               <xsl:apply-templates select=".">
                  <xsl:with-param name="file_name" select="$file_name" />
               </xsl:apply-templates>
            </xsl:for-each>
         </xsl:if>
      </ul>
      <br/>           
   </xsl:template>
   
   <xsl:template match="top">
      <xsl:param name="file_name" />
      <xsl:apply-templates>
         <xsl:with-param name="file_name" select="$file_name" /> 
      </xsl:apply-templates>
   </xsl:template>
   
   <xsl:template match="bottom">
      <xsl:param name="file_name" />
      <xsl:apply-templates>
         <xsl:with-param name="file_name" select="$file_name" />
      </xsl:apply-templates>
   </xsl:template>
   
   <xsl:template match="download_button">
      <xsl:variable name="linktext"><xsl:apply-templates select="link"/></xsl:variable>
      <a href="{$linktext}" id="download_button" class="menu"><xsl:apply-templates select="name"/></a>
   </xsl:template>
   
   
   <xsl:template match="menu">
      <xsl:param name="file_name" />
      <div class="menu_top">
         <xsl:apply-templates select="top">
            <xsl:with-param name="file_name" select="$file_name" />
         </xsl:apply-templates>
      </div>
      <div class="menu_footer">
         <xsl:apply-templates select="bottom">
            <xsl:with-param name="file_name" select="$file_name" />
         </xsl:apply-templates>
      </div>
   </xsl:template>
   
   <xsl:template match="item">
      <xsl:param name="file_name" />
      <li>
      <xsl:choose>
         <xsl:when test="@nolink = 'true'">
            <xsl:choose>
               <xsl:when test="name">
                  <a onclick="Toggle(this)" class="sub menu"><img src="plus.gif"/><xsl:value-of select="name"/></a>
                  <xsl:apply-templates select="sub">
                     <xsl:with-param name="file_name" select="$file_name" />
                  </xsl:apply-templates> 
               </xsl:when>
               <xsl:otherwise>
                  <xsl:apply-templates>
                      <xsl:with-param name="file_name" select="$file_name" />
                  </xsl:apply-templates>
               </xsl:otherwise>
            </xsl:choose>           
         </xsl:when>
         <xsl:when test="name">
            <xsl:choose>
               <xsl:when test="sub">
                  <xsl:choose>
                     <xsl:when test="link">
                        <xsl:variable name="linktext"><xsl:apply-templates select="link"/></xsl:variable>
                        <a href="{$linktext}" style="float:right"><img src="right.gif" border="0"/></a>
                        <a onclick="Toggle(this)" class="sub menu"><img src="plus.gif" border="0"/><xsl:value-of select="name"/></a>
                        <xsl:apply-templates select="sub">
                           <xsl:with-param name="file_name" select="$file_name" />
                        </xsl:apply-templates> 
                     </xsl:when>
                     <xsl:otherwise>
                        <a href="{$file_name}#{name}" style="float:right"><img src="down.gif" border="0" /></a>
                        <a onclick="Toggle(this)" class="sub menu"><img src="plus.gif" border="0"/><xsl:value-of select="name"/></a>
                        <xsl:apply-templates select="sub">
                           <xsl:with-param name="file_name" select="$file_name" />
                        </xsl:apply-templates>
                     </xsl:otherwise>
                  </xsl:choose>
               </xsl:when>
               <xsl:otherwise>
                  <xsl:choose>
                     <xsl:when test="link">
                        <xsl:variable name="linktext"><xsl:apply-templates select="link"/></xsl:variable>
                        <a href="{$linktext}" class="menu"><xsl:value-of select="name"/></a>
                     </xsl:when>
                     <xsl:otherwise>
                        <a href="{$file_name}#{name}" class="menu"><xsl:value-of select="name"/></a>
                     </xsl:otherwise>
                  </xsl:choose>
               </xsl:otherwise>
            </xsl:choose>
         </xsl:when>
         <xsl:otherwise>
            <a href="{$file_name}#{.}" class="menu"><xsl:value-of select="."/></a>
         </xsl:otherwise>
      </xsl:choose>
  </li>
   </xsl:template>   
   
   <xsl:template match="sub">
      <xsl:param name="file_name" />
      <ul  style="display:none;">
         <xsl:for-each select="item">  
         <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/> 
            <xsl:apply-templates select=".">
               <xsl:with-param name="file_name" select="$file_name" />
            </xsl:apply-templates>
         </xsl:for-each>
      </ul>
   </xsl:template>   
   
   
   <!-- ************************************************************************* -->
      
   <xsl:template match="questions">

      <xsl:for-each select="question">      
         <xsl:sort select="translate(@text,$lcletters, $ucletters)"/> 

         <a name = "{@text}"/>
         <a name = "{translate(@text,$badletters,'')}">
            <div class="question">
               <a href="#top"><font size='2'><center>[top]</center></font></a>
               <h2><xsl:value-of select="@text"/></h2>
               <xsl:apply-templates select="."/>
            </div>
         </a>
      </xsl:for-each>
   </xsl:template>
      
   <!-- ************************************************************************* -->
      
   <xsl:template match="components">

      <xsl:for-each select="component">      
      <xsl:sort select="translate(name,$lcletters, $ucletters)"/> 
         <xsl:variable name="checked" select="@checked"/>

         <a name = "{name}"/>
         <div class="component"  >
      
         <a href="#top"><font size='2'><center>[top]</center></font></a>
         <h1 style="margin:0px;"><xsl:value-of select="name"/> 
         </h1>
         <BR/>
         <BR/>
         <xsl:apply-templates select="description"/>

         <BR/>
         <xsl:apply-templates select="examples"/>
         <div class='include_file_more_details_wrapper'>
            <xsl:if test="spec_file">
               <xsl:choose>
                  <xsl:when test="spec_file/@link = 'true'">
                     <a class='more_details' href="{spec_file}.html#{name}">More Details...</a> 
                  </xsl:when>
                  <xsl:otherwise>
                     <a class='more_details' href="{spec_file}.html">More Details...</a>
                  </xsl:otherwise>
               </xsl:choose>
            </xsl:if>
            <xsl:if test="file">
               <div class='include_file'>#include &lt;<xsl:value-of select="file"/>&gt;</div>
            </xsl:if>
         </div>

         <xsl:apply-templates select="implementations">           
            <xsl:with-param name="checked" select="$checked" />
         </xsl:apply-templates>
               
      
         <xsl:if test="extensions">
            <br/>
            <center>
            <h1>Extensions to <xsl:value-of select="name"/></h1>
            </center>
            
            <xsl:for-each select="extensions/extension">
            <xsl:sort select="translate(name,$lcletters, $ucletters)"/> 
               <div class="extension">
               <a name="{name}"><B><font size='5'><xsl:value-of select="name"/></font></B></a><Br/>
               <BR/>
               <xsl:apply-templates select="description"/>
               <BR/>
               <BR/>
               <xsl:apply-templates select="examples"/>
               <xsl:choose>
                  <xsl:when test="spec_file/@link = 'true'">
                     <a class='more_details_extension' href="{spec_file}.html#{name}">More Details...</a>
                  </xsl:when>
                  <xsl:otherwise>
                     <a class='more_details_extension' href="{spec_file}.html">More Details...</a>
                  </xsl:otherwise>
               </xsl:choose>


               <xsl:apply-templates select="implementations">           
                  <xsl:with-param name="checked" select="$checked" />
               </xsl:apply-templates>
               </div>
            </xsl:for-each>            
         </xsl:if>
      
         </div>
      </xsl:for-each>
   </xsl:template>      

   <!-- This template outputs a length 1 string if there is a python example program -->
   <xsl:template name="has_python_example">
      <xsl:for-each select="example">
         <xsl:if test="substring-before(.,'.py.html') != ''">1</xsl:if>
      </xsl:for-each>
   </xsl:template>
   <!-- This template outputs a length 1 string if there is a C++ example program -->
   <xsl:template name="has_cpp_example">
      <xsl:for-each select="example">
         <xsl:if test="substring-before(.,'.cpp.html') != ''">1</xsl:if>
      </xsl:for-each>
   </xsl:template>

   <xsl:template match="examples">
      <xsl:variable name="has_python"><xsl:call-template name="has_python_example"/></xsl:variable>
      <xsl:variable name="has_cpp"><xsl:call-template name="has_cpp_example"/></xsl:variable>
      <xsl:variable name="numpy" select="string-length($has_python)"/>
      <xsl:variable name="numcpp" select="string-length($has_cpp)"/>

      <xsl:if test="$numcpp != 0"> <BR/>C++ Example Programs: </xsl:if>
      <xsl:for-each select="example">
         <xsl:variable name="fname" select="substring-before(.,'.cpp.html')"/>
         <xsl:variable name="name" select="substring-before(.,'.html')"/>
         <xsl:if test="$fname != ''">
            <xsl:choose>
               <xsl:when test="position() >= last()-$numpy">
                  <a href="{.}"><xsl:value-of select="$name"/></a>
               </xsl:when>
               <xsl:otherwise>
                  <a href="{.}"><xsl:value-of select="$name"/></a>,
               </xsl:otherwise>
            </xsl:choose>              
         </xsl:if>
      </xsl:for-each>

      <xsl:if test="$numpy != 0"> <BR/>Python Example Programs: </xsl:if>
      <xsl:for-each select="example">
         <xsl:variable name="fname" select="substring-before(.,'.py.html')"/>
         <xsl:variable name="name" select="substring-before(.,'.html')"/>
         <xsl:if test="$fname != ''">
            <xsl:choose>
               <xsl:when test="position() >= last()">
                  <a href="{.}"><xsl:value-of select="$name"/></a>
               </xsl:when>
               <xsl:otherwise>
                  <a href="{.}"><xsl:value-of select="$name"/></a>,
               </xsl:otherwise>
            </xsl:choose>              
         </xsl:if>
      </xsl:for-each>
   </xsl:template>

   <xsl:template match="implementations">
      <xsl:param name="checked" />
      <BR/><BR/><B>Implementations:</B>
      
      <xsl:choose>
         <xsl:when test="implementation/typedefs">
            <xsl:for-each select="implementation">
               <blockquote>
                  <a href="{file}.html"><xsl:value-of select="name"/></a>:
                  <xsl:if test="typedefs"><br/></xsl:if>
                  <xsl:apply-templates select="description"/>
                  <xsl:apply-templates select="typedefs">               
                     <xsl:with-param name="checked" select="$checked"/>
                  </xsl:apply-templates>
               </blockquote>     
            </xsl:for-each>         
         </xsl:when>
         <xsl:otherwise>
            <blockquote>
               <xsl:for-each select="implementation">
                  <a href="{file}.html"><xsl:value-of select="name"/></a>:
                  <xsl:apply-templates select="description"/>
                  <br/>
               </xsl:for-each>      
            </blockquote>        
         </xsl:otherwise>
      </xsl:choose>
   </xsl:template>   
   
   <xsl:template match="typedefs">
      <xsl:param name="checked" />
      
   
      <div class="typedefs"><table CELLSPACING='0' CELLPADDING='0' bgcolor="white" >       
         <xsl:for-each select="typedef">
         
            <xsl:choose>
               <xsl:when test="$checked = 'true'">
                  <tr><td bgcolor="{$gray}" valign="top"> 
                  <div class="tdn"><xsl:value-of select="name"/></div>  
                  </td><td width="100%" bgcolor="{$gray}"> 
                  <xsl:apply-templates select="description"/>
                  </td></tr>                 
               
                  <tr><td valign="top"> 
                  <div class="tdn"><xsl:value-of select="name"/>_c</div>
                  </td><td width="100%"> 
                  is a typedef for <xsl:value-of select="name"/> that checks its preconditions.             
                  </td></tr>                 
               </xsl:when>
               <xsl:otherwise>
                  <xsl:choose>
                     <xsl:when test="position() mod 2 = 0">
                        <tr><td valign="top"> 
                        <div class="tdn"><xsl:value-of select="name"/></div>  
                        </td><td width="100%"> 
                        <xsl:apply-templates select="description"/>
                        </td></tr>                                
                     </xsl:when>
                     <xsl:otherwise>
                        <tr><td bgcolor="{$gray}" valign="top"> 
                        <div class="tdn"><xsl:value-of select="name"/></div>  
                        </td><td width="100%" bgcolor="{$gray}"> 
                        <xsl:apply-templates select="description"/>
                        </td></tr>  
                     </xsl:otherwise>
                  </xsl:choose>              
               </xsl:otherwise>           
            </xsl:choose>
         </xsl:for-each>      
      </table></div>
   </xsl:template>
   
   <!-- ************************************************************************* -->

   <xsl:template match="release_notes">
         <h1 style="margin:0px;">Release <xsl:value-of select="$current_release_var"/></h1>
               <u>Release date</u>: <xsl:value-of select="$last_modified_date_var"/>
               <br/>
               <u>Major Changes in this Release</u>:
            <table cellspacing="5" cellpadding="0" width="100%">
               <tr>
                  <td width="15"></td>
                  <td><pre><xsl:value-of select="current"/></pre></td>
               </tr>
            </table>
   
      <xsl:for-each select="old">
         <xsl:if test="position() &lt; 10">
         <hr/>
         <h1 style="margin:0px;">Release <xsl:value-of select="@name"/></h1>
            <xsl:if test="@date">
               <u>Release date</u>: <xsl:value-of select="@date"/>
            </xsl:if>
            <br/>
               <u>Major Changes in this Release</u>:
            <table cellspacing="5" cellpadding="0" width="100%">
               <tr>
                  <td width="15"></td>
                  <td><pre><xsl:value-of select="."/></pre></td>
               </tr>
            </table>
         </xsl:if>
      </xsl:for-each>
      <br/>
      <br/>
      <br/>
      <center><a href="old_release_notes.html">Old Release Notes</a></center>
      <br/>

   </xsl:template>      

   <!-- ************************************************************************* -->

   <xsl:template match="old_release_notes">
      <xsl:for-each select="document('release_notes.xml')/doc/body/release_notes/old">
         <xsl:if test="position() &gt;= 10">
         <h1 style="margin:0px;">Release <xsl:value-of select="@name"/></h1>
            <xsl:if test="@date">
               <u>Release date</u>: <xsl:value-of select="@date"/>
            </xsl:if>
            <br/>
               <u>Major Changes in this Release</u>:
            <table cellspacing="5" cellpadding="0" width="100%">
               <tr>
                  <td width="15"></td>
                  <td><pre><xsl:value-of select="."/></pre></td>
               </tr>
            </table>
            <xsl:if test="position() != last()">
               <hr/>
            </xsl:if>
         </xsl:if>
      </xsl:for-each>
   </xsl:template>      

   <!-- ************************************************************************* -->

   <xsl:template match="description">
      <xsl:apply-templates/>
   </xsl:template>      
         
   <xsl:template match="body">
       <xsl:choose>
           <xsl:when test="@from_file">
               <xsl:apply-templates select="document(@from_file)"/>            
               <xsl:apply-templates/>
         </xsl:when>
           <xsl:otherwise>
               <xsl:apply-templates/>
           </xsl:otherwise>
       </xsl:choose>
   </xsl:template>

   <xsl:template match="text">
      <br/>
      <pre><xsl:apply-templates/></pre>
   </xsl:template>
   
      
   <xsl:template match="h3">
      <a name="{.}"/>
      <h3>
         <xsl:apply-templates/>
       </h3>
   </xsl:template>
   
   <xsl:template match="h2">
      <a name="{.}"/>
      <h2>
         <xsl:apply-templates/>
       </h2>
   </xsl:template>
      
   <xsl:template match="h1">
      <a name="{.}"/>
      <h1>
         <xsl:apply-templates/>
       </h1>
   </xsl:template>

   <xsl:template match="preserve_space">
         <xsl:apply-templates/>
   </xsl:template>
   
   <xsl:template match="p">
      <p>
         <xsl:apply-templates/>
       </p>
   </xsl:template>   
   <xsl:template match="center">
      <center>
         <xsl:apply-templates/>
       </center>
   </xsl:template>   
   <xsl:template match="pre">
      <pre>
         <xsl:apply-templates/>
       </pre>
   </xsl:template>   
   <xsl:template match="blockquote">
      <blockquote>
         <xsl:apply-templates/>
       </blockquote>
   </xsl:template>   
   <xsl:template match="anchor">    
      <a name="{.}"/>
   </xsl:template>   
   <xsl:template match="br">
      <br/>
         <xsl:apply-templates/>
   </xsl:template>   

   <xsl:template match="youtube">
      <iframe width="900" height="506" src="{@src}" frameborder="0" allowfullscreen='1'></iframe>
   </xsl:template>   

   <xsl:template match="a">
      <a href="{@href}">
         <xsl:apply-templates/>
       </a>
   </xsl:template>   
   <xsl:template match="script">
      <script type="{@type}" language="{@language}" src="{@src}">
         <xsl:apply-templates/>
       </script>
   </xsl:template>   
   <xsl:template match="tt">
      <tt>
         <xsl:apply-templates/>
       </tt>
   </xsl:template>   
   <xsl:template match="chm">
      <xsl:if test="$is_chm = 'true'">
         <xsl:apply-templates/>
       </xsl:if>
   </xsl:template>   
   <xsl:template match="web">
      <xsl:if test="$is_chm != 'true'">
         <xsl:apply-templates/>
       </xsl:if>
   </xsl:template>   
   <xsl:template match="td">
      <xsl:if test="@colspan">
         <td align="center" colspan="{@colspan}">
            <xsl:apply-templates/>
         </td>
      </xsl:if>
      <xsl:if test="not(@colspan)">
         <td align="center">
            <xsl:apply-templates/>
         </td>
      </xsl:if>
   </xsl:template>   
   <xsl:template match="tr">
      <tr>
         <xsl:apply-templates/>
       </tr>
   </xsl:template>   
   <xsl:template match="table">
      <table>
         <xsl:apply-templates/>
       </table>
   </xsl:template>   
   <xsl:template match="more_details">
      <a style="float:none" class='more_details'>More Details...</a>
   </xsl:template>   
   <xsl:template match="div">
      <div id="{@id}"><xsl:apply-templates/></div>
   </xsl:template>   
   <xsl:template match="li">
      <li>
         <xsl:apply-templates/>
       </li>
   </xsl:template>   
   <xsl:template match="ul">
      <xsl:if test="@style">
         <ul style="{@style}">
            <xsl:apply-templates/>
         </ul>
      </xsl:if>
      <xsl:if test="not(@style)">
         <ul>
            <xsl:apply-templates/>
         </ul>
      </xsl:if>
   </xsl:template>   
   <xsl:template match="ol">
      <ol>
         <xsl:apply-templates/>
       </ol>
   </xsl:template>   
   <xsl:template match="u">
      <u>
         <xsl:apply-templates/>
       </u>
   </xsl:template>   
   <xsl:template match="i">
      <i>
         <xsl:apply-templates/>
       </i>
   </xsl:template>   
   <xsl:template match="b">
      <b>
         <xsl:apply-templates/>
       </b>
   </xsl:template>   
   <xsl:template match="U">
      <u>
         <xsl:apply-templates/>
       </u>
   </xsl:template>   
   <xsl:template match="I">
      <i>
         <xsl:apply-templates/>
       </i>
   </xsl:template>   
   <xsl:template match="B">
      <b>
         <xsl:apply-templates/>
       </b>
   </xsl:template>   
   <xsl:template match="font">
      <xsl:if test="@style">
         <font color="{@color}" style="{@style}">
            <xsl:apply-templates/>
         </font>
      </xsl:if>
      <xsl:if test="not(@style)">
         <font color="{@color}">
            <xsl:apply-templates/>
         </font>
      </xsl:if>
   </xsl:template>   
   <xsl:template match="image">
      <img src="{@src}" border="0"/>
   </xsl:template>   
   <xsl:template match="img">
      <img src="{@src}" border="0" height="{@height}" width="{@width}" alt="{@alt}">
         <xsl:apply-templates/>
       </img>
   </xsl:template>   
   <xsl:template match="video">
      <video controls="true" poster="{@src}.png">
         <source src="{@src}.webm" type="video/webm"/>
         <source src="{@src}.mp4" type="video/mp4"/>
         <xsl:apply-templates/>
      </video>
   </xsl:template>   

   <xsl:template name="term_list_go">
      <xsl:param name="num"/>
      <xsl:if test="$num &lt; 27">
               <xsl:variable name="cur_letter" select="substring($ucletters, $num, 1)"/>

               <div style="padding:1em"> 
               <div style="display: inline-block;width:100% ">
               <a name="{$cur_letter}"/>
               
               <h1><xsl:value-of select="$cur_letter"/></h1>
               <xsl:for-each select="term">
               <xsl:sort order="ascending" select="translate(@name,$lcletters, $ucletters)"/>
               <xsl:variable name="alt" select="1+(position() mod 2)"/>
               <xsl:variable name="line" select="concat('line',format-number($alt,'0'))"/>
               <xsl:if test="$cur_letter = substring(translate(@name,$lcletters, $ucletters),1,1)">
               <xsl:choose>
                  <xsl:when test="@link">
                     <div class='{$line}'><div class='name'><a href="{@link}"><xsl:value-of select="@name"/></a></div>
                     <div class='inc'><xsl:if test='@include'><b>#include &lt;<xsl:value-of select="@include"/>&gt;</b></xsl:if></div>
                     </div>
                  </xsl:when>
                  <xsl:when test="@file">
                     <div class='{$line}'><div class='name'><a href="{@file}#{@name}"><xsl:value-of select="@name"/></a></div>
                     <div class='inc'><xsl:if test='@include'><b>#include &lt;<xsl:value-of select="@include"/>&gt;</b></xsl:if></div>
                     </div>
                  </xsl:when>
                  <xsl:otherwise>
                      <xsl:value-of select="@name"/>
                      <div style="padding-left: 50px;">
                        <xsl:for-each select="term">
                        <xsl:sort order="ascending" select="translate(@name,$lcletters, $ucletters)"/> 
                        <xsl:variable name="alt2" select="1+(($alt+position()) mod 2)"/>
                        <xsl:variable name="line2" select="concat('line',format-number($alt2,'0'))"/>
                           <div class='{$line2}'><div class='name'><a href="{@link}"><xsl:value-of select="@name"/></a></div>
                           <div class='inc'><xsl:if test='@include'><b>#include &lt;<xsl:value-of select="@include"/>&gt;</b></xsl:if></div>
                           </div>
                        </xsl:for-each>
                      </div>
                  </xsl:otherwise>
               </xsl:choose>
               </xsl:if>
               </xsl:for-each>
               </div>
               </div>

      <xsl:call-template name="term_list_go" >
         <xsl:with-param name="num" select="$num + 1"/>
      </xsl:call-template>

      </xsl:if>
   </xsl:template>   


   <xsl:template match="term_list">
      <center>
         <div style="font-size:1.2em">
      <a href="#A">[A]</a> 
      <a href="#B">[B]</a> 
      <a href="#C">[C]</a> 
      <a href="#D">[D]</a> 
      <a href="#E">[E]</a> 
      <a href="#F">[F]</a> 
      <a href="#G">[G]</a> 
      <a href="#H">[H]</a> 
      <a href="#I">[I]</a> 
      <a href="#J">[J]</a> 
      <a href="#K">[K]</a> 
      <a href="#L">[L]</a> 
      <a href="#M">[M]</a> 
      <a href="#N">[N]</a> 
      <a href="#O">[O]</a> 
      <a href="#P">[P]</a> 
      <a href="#Q">[Q]</a> 
      <a href="#R">[R]</a> 
      <a href="#S">[S]</a> 
      <a href="#T">[T]</a> 
      <a href="#U">[U]</a> 
      <a href="#V">[V]</a> 
      <a href="#W">[W]</a> 
      <a href="#X">[X]</a> 
      <a href="#Y">[Y]</a> 
      <a href="#Z">[Z]</a> 
         </div>
      </center>
      <xsl:call-template name="term_list_go" >
         <xsl:with-param name="num" select="1"/>
      </xsl:call-template>
   </xsl:template>   



   <!--  This function turns a string of the form 2006-03-21T02:35:20+00:00 into a nice 
   normal looking date -->
   <xsl:template name="format-date">
      <xsl:param name="xsd-date"/>
      <xsl:variable name="date" select="substring-before($xsd-date,'T')"/>
      <xsl:variable name="time" select="substring-after($xsd-date,'T')"/>

      <xsl:variable name="year" select="substring($date,1,4)"/>
      <xsl:variable name="month" select="substring($date,6,2)"/>
      <xsl:variable name="day" select="substring($date,9,2)"/>
      <xsl:variable name="lhour" select="substring($time,1,2)"/>
      <xsl:variable name="lminute" select="substring($time,4,2)"/>
      <xsl:variable name="second" select="substring($time,7,2)"/>

      <xsl:variable name="ohour" select="substring($time,10,2)"/>
      <xsl:variable name="ominute" select="substring($time,13,2)"/>


    <xsl:choose>
      <xsl:when test="$month = 1">Jan </xsl:when>
      <xsl:when test="$month = 2">Feb </xsl:when>
      <xsl:when test="$month = 3">Mar </xsl:when>
      <xsl:when test="$month = 4">Apr </xsl:when>
      <xsl:when test="$month = 5">May </xsl:when>
      <xsl:when test="$month = 6">Jun </xsl:when>
      <xsl:when test="$month = 7">Jul </xsl:when>
      <xsl:when test="$month = 8">Aug </xsl:when>
      <xsl:when test="$month = 9">Sep </xsl:when>
      <xsl:when test="$month = 10">Oct </xsl:when>
      <xsl:when test="$month = 11">Nov </xsl:when>
      <xsl:when test="$month = 12">Dec </xsl:when>
    </xsl:choose>

      <xsl:variable name="op" select="substring($time,9,1)"/>
      <xsl:if test="$op = '-'">
         <xsl:variable name="hour"   select="format-number(number($lhour)-number($ohour),'00')"/>
         <xsl:variable name="minute" select="format-number(number($lminute)-number($ominute),'00')"/>
         <xsl:value-of select="$day"/>, <xsl:value-of select="$year"/>
         (<xsl:value-of select="$hour"/>:<xsl:value-of select="$minute"/>:<xsl:value-of select="$second"/> UTC)
      </xsl:if>
      <xsl:if test="$op = '+'">
         <xsl:variable name="hour"   select="format-number(number($lhour)+number($ohour),'00')"/>
         <xsl:variable name="minute" select="format-number(number($lminute)+number($ominute),'00')"/>
         <xsl:value-of select="$day"/>, <xsl:value-of select="$year"/>
         (<xsl:value-of select="$hour"/>:<xsl:value-of select="$minute"/>:<xsl:value-of select="$second"/> UTC)
      </xsl:if>

   </xsl:template>



   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->
   <!-- *******    Subversion stylesheet stuff   ******************** -->
   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->


   
   <!-- ************************************************************************* -->
   
   
   <xsl:template match="log">
      <xsl:for-each select="logentry">
      <xsl:sort order="descending" data-type="number" select="./@revision"/>
      <u>Revision</u>: <xsl:value-of select="substring(@revision,1,20)"/> <br/>
      <u>Author</u>: <a href="mailto:{author/@email}"><xsl:value-of select="author"/></a> <br/>
      <u>Date</u>: <xsl:call-template name="format-date"><xsl:with-param name="xsd-date" select="date"/></xsl:call-template> <br/>
            <xsl:apply-templates select="msg"/>
            <xsl:apply-templates select="paths"/>
            <pre class="files_changed"><xsl:value-of select="files_changed"/></pre>
      <hr class="fullhr"/>
      </xsl:for-each>
   </xsl:template>
   
         


   
   <!-- ************************************************************************* -->

   <xsl:template name="paths">
    <xsl:param name="type"/>
    <xsl:param name="name"/>
    <xsl:param name="color"/>

     <xsl:if test="path[@action=$type]">

      <div class="logb">
         <div class="bsolid"><b><font color="{$color}"><xsl:value-of select="$name"/></font></b></div>
            <xsl:for-each select="path[@action = $type]">
            <xsl:sort select="."/>
               <xsl:choose>
                  <xsl:when test="position() mod 2 = 0">
                     <div class="row1"><xsl:value-of select="."/></div>
                  </xsl:when>
                  <xsl:otherwise>
                     <div class="row2"><xsl:value-of select="."/></div>
                  </xsl:otherwise>
               </xsl:choose>
            </xsl:for-each>
      </div>
     </xsl:if>
   </xsl:template>

   <!-- ************************************************************************* -->
   <xsl:template match="paths">
      <xsl:call-template name="paths">
         <xsl:with-param name="type">M</xsl:with-param>
         <xsl:with-param name="name">Modified</xsl:with-param>
         <xsl:with-param name="color">black</xsl:with-param>
      </xsl:call-template>
      <xsl:call-template name="paths">
         <xsl:with-param name="type">A</xsl:with-param>
         <xsl:with-param name="name">Added</xsl:with-param>
         <xsl:with-param name="color">blue</xsl:with-param>
      </xsl:call-template>
      <xsl:call-template name="paths">
         <xsl:with-param name="type">D</xsl:with-param>
         <xsl:with-param name="name">Deleted</xsl:with-param>
         <xsl:with-param name="color">red</xsl:with-param>
      </xsl:call-template>
      <xsl:call-template name="paths">
         <xsl:with-param name="type">R</xsl:with-param>
         <xsl:with-param name="name">Deleted</xsl:with-param>
         <xsl:with-param name="color">red</xsl:with-param>
      </xsl:call-template>
   </xsl:template>

   <xsl:template match="msg">
      <p style="margin:0.4em"><xsl:value-of select="."/></p>
   </xsl:template>

   
   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->
   <!-- XSLT for dealing with <code> blocks generated by the htmlify to-xml option -->
   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->

   <xsl:template match="code">

      <h1>Classes and Structs:</h1>
      <xsl:for-each select="classes/class">  
         <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/>  
         <xsl:apply-templates select="."/>
      </xsl:for-each>

      <h1>Global Functions:</h1>
      <xsl:for-each select="global_functions/function">  
         <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/>  
         <div class="function">
            <a onclick="Toggle(this)" style="cursor: pointer"><img src="plus.gif" border="0"/><font color="blue">
               <u><b><xsl:value-of select="name"/>()</b></u></font></a>
            <div style="display:none;">
               <br/>
               <xsl:if test="scope != ''">
                  <u>Scope</u>: <xsl:value-of select="scope"/> <br/>
               </xsl:if>
               <u>File</u>: <xsl:value-of select="file"/> <br/><br/>
               <div style="margin-left:1.5em">
               <pre style="font-size:1.1em;"><xsl:value-of select="declaration"/>;</pre> 
               <font color="#009900"><pre><xsl:value-of select="comment"/></pre></font> 
               </div>
               <br/>
            </div>
         </div>
      </xsl:for-each>

   </xsl:template>

   <!-- ************************************************************************* -->

   <xsl:template match="class">
         <div class="class">
            <a onclick="Toggle(this)" style="cursor: pointer"><img src="plus.gif" border="0"/><font color="blue">
               <u><b><xsl:value-of select="name"/></b></u></font></a>
            <div style="display:none;">
               <br/>
               <xsl:if test="scope != ''">
                  <u>Scope</u>: <xsl:value-of select="scope"/> <br/>
               </xsl:if>
               <u>File</u>: <xsl:value-of select="file"/> <br/><br/>
               <div style="margin-left:1.5em">
               <pre style="font-size:1.1em;"><xsl:value-of select="declaration"/>;</pre> <br/>
               <font color="#009900"><pre><xsl:value-of select="comment"/></pre></font> <br/>
               </div>

               <xsl:if test="protected_typedefs">
                  <a onclick="BigToggle(this)" style="cursor: pointer"><img src="bigplus.gif" border="0"/><font color="blue">
                     <u style="font-size:2em">Protected Typedefs</u></font></a>
                     <div style="display:none;">
                        <ul>
                           <xsl:for-each select="protected_typedefs/typedef">
                              <li><xsl:value-of select="."/>;</li>
                           </xsl:for-each>
                        </ul>
                     </div>
                  <br/>
               </xsl:if>

               <xsl:if test="public_typedefs">
                  <a onclick="BigToggle(this)" style="cursor: pointer"><img src="bigplus.gif" border="0" style="size:2em"/><font color="blue">
                     <u style="font-size:2em">Public Typedefs</u></font></a>
                     <div style="display:none;">
                        <ul>
                           <xsl:for-each select="public_typedefs/typedef">
                              <li><xsl:value-of select="."/>;</li>
                           </xsl:for-each>
                        </ul>
                     </div>
                  <br/>
               </xsl:if>

               <xsl:if test="protected_variables">
                  <a onclick="BigToggle(this)" style="cursor: pointer"><img src="bigplus.gif" border="0"/><font color="blue">
                     <u style="font-size:2em">Protected Variables</u></font></a>
                  <div style="display:none;">
                     <ul>
                        <xsl:for-each select="protected_variables/variable">
                           <li><xsl:value-of select="."/>;</li>
                        </xsl:for-each>
                     </ul>
                  </div>
                  <br/>
               </xsl:if>

               <xsl:if test="public_variables">
                  <a onclick="BigToggle(this)" style="cursor: pointer"><img src="bigplus.gif" border="0"/><font color="blue">
                     <u style="font-size:2em">Public Variables</u></font></a>
                  <div style="display:none;">
                     <ul>
                        <xsl:for-each select="public_variables/variable">
                           <li><xsl:value-of select="."/>;</li>
                        </xsl:for-each>
                     </ul>
                  </div>
                  <br/>
               </xsl:if>

               <xsl:if test="protected_methods">
                  <a onclick="BigToggle(this)" style="cursor: pointer"><img src="bigplus.gif" border="0"/><font color="blue">
                     <u style="font-size:2em">Protected Methods</u></font></a>
                  <div style="display:none;">
                  <xsl:for-each select="protected_methods/method">
                     <div class="function">
                        <u>Method Name</u>: <b><xsl:value-of select="name"/></b> <br/><br/>
                        <div style="margin-left:1.5em">
                           <pre style="font-size:1.1em;"><xsl:value-of select="declaration"/>;</pre> 
                           <font color="#009900"><pre><xsl:value-of select="comment"/></pre></font> <br/>
                        </div>
                     </div>
                  </xsl:for-each>
                  </div>
                  <br/>
               </xsl:if>

               <xsl:if test="public_methods">
                  <a onclick="BigToggle(this)" style="cursor: pointer"><img src="bigplus.gif" border="0"/><font color="blue">
                     <u style="font-size:2em">Public Methods</u></font></a>
                  <div style="display:none;">
                  <xsl:for-each select="public_methods/method">
                     <div class="function">
                        <u>Method Name</u>: <b><xsl:value-of select="name"/></b> <br/><br/>
                        <div style="margin-left:1.5em">
                           <pre style="font-size:1.1em;"><xsl:value-of select="declaration"/>;</pre> 
                           <font color="#009900"><pre><xsl:value-of select="comment"/></pre></font> <br/>
                        </div>
                     </div>
                  </xsl:for-each>
                  </div>
                  <br/>
               </xsl:if>

               <xsl:if test="protected_inner_classes">
                  <a onclick="BigToggle(this)" style="cursor: pointer"><img src="bigplus.gif" border="0"/><font color="blue">
                     <u style="font-size:2em">Protected Inner Classes</u></font></a>
                  <div style="display:none;">
                  <xsl:for-each select="protected_inner_classes/class">
                     <xsl:apply-templates select="."/>
                  </xsl:for-each>
                  </div>
                  <br/>
               </xsl:if>

               <xsl:if test="public_inner_classes">
                  <a onclick="BigToggle(this)" style="cursor: pointer"><img src="bigplus.gif" border="0"/><font color="blue">
                     <u style="font-size:2em">Public Inner Classes</u></font></a>
                  <div style="display:none;">
                  <xsl:for-each select="public_inner_classes/class">
                     <xsl:apply-templates select="."/>
                  </xsl:for-each>
                  </div>
                  <br/>
               </xsl:if>

            </div>
         </div>
   </xsl:template>

   <!-- ************************************************************************* -->

   <xsl:template match="code_box">
      <pre class="code_box"><xsl:apply-templates/></pre>
   </xsl:template>  

   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->



   
</xsl:stylesheet>
