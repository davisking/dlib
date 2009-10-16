<?xml version="1.0" encoding="ISO-8859-1" ?>
   
<!--
   To the extent possible under law, Davis E King  has waived all copyright and 
   related or neighboring rights to dlib documentation (XML, HTML, and XSLT files).
   This work is published from United States. 
-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
   <xsl:output method='html' version='1.0' encoding='UTF-8' indent='yes' />
   
   
   <!-- ************************************************************************* -->
   
   <xsl:variable name="is_chm">true</xsl:variable>
   <xsl:variable name="main_menu">main_menu.xml</xsl:variable>
   <xsl:variable name="project_name">dlib C++ Library</xsl:variable>
   
   <!-- ************************************************************************* -->

   <!-- This is the ID you get from Google Webmaster Tools -->
   <xsl:variable name="google_verify_id">02MiiaFNVzS5/u0eQhsy3/knioFHsia1X3DXRpHkE6I=</xsl:variable>

   <!-- ************************************************************************* -->

   <xsl:variable name="last_modified_date_var">_LAST_MODIFIED_DATE_</xsl:variable>
   <xsl:variable name="current_release_var">_CURRENT_RELEASE_</xsl:variable>
   <xsl:template match="last_modified_date"><xsl:value-of select="$last_modified_date_var"/></xsl:template>
   <xsl:template match="current_release"><xsl:value-of select="$current_release_var"/></xsl:template>
   
   <!-- ************************************************************************* -->
   
   <xsl:variable name="gray">#E3E3E3</xsl:variable>
   <xsl:variable name="background_color">#EDF3EE</xsl:variable>
   <xsl:variable name="main_width">62.5em</xsl:variable>

   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->

   <xsl:variable name="lcletters">abcdefghijklmnopqrstuvwxyz </xsl:variable>
   <xsl:variable name="ucletters">ABCDEFGHIJKLMNOPQRSTUVWXYZ </xsl:variable>
   
   <!-- ************************************************************************* -->

   <xsl:template match="/doc">
      <html>
         <head>
            <!-- Verify with Google -->
            <meta name="verify-v1" content="{$google_verify_id}" />
            <title>
               <xsl:value-of select="$project_name"/>
               <xsl:if test="title">
               - <xsl:value-of select="title" />
               </xsl:if>
            </title>


      <xsl:if test="$is_chm != 'true'">
            <!-- Piwik -->
            <script type="text/javascript">
            var pkBaseURL = (("https:" == document.location.protocol) ? "https://apps.sourceforge.net/piwik/dclib/" : "http://apps.sourceforge.net/piwik/dclib/");
            document.write(unescape("%3Cscript src='" + pkBaseURL + "piwik.js' type='text/javascript'%3E%3C/script%3E"));
            </script><script type="text/javascript">
            piwik_action_name = '';
            piwik_idsite = 1;
            piwik_url = pkBaseURL + "piwik.php";
            piwik_log(piwik_action_name, piwik_idsite, piwik_url);
            </script>
            <object><noscript><p><img src="http://apps.sourceforge.net/piwik/dclib/piwik.php?idsite=1" alt="piwik"/></p></noscript></object>
            <!-- End Piwik Tag -->
       </xsl:if>



            <!-- [client side code for collapsing and unfolding branches] -->
            <script language="JavaScript">

            // ---------------------------------------------
            // --- Name:    Easy DHTML Treeview           --
            // --- Author:  D.D. de Kerf                  --
            // --- Version: 0.2          Date: 13-6-2001  --
            // ---------------------------------------------
            function Toggle(node)
            {
               // Unfold the branch if it isn't visible
               var next_node = node.nextSibling;
               if (next_node.style.display == 'none')
               {
                  // Change the image (if there is an image)
                  if (node.childNodes.length > 0)
                  {
                     if (node.childNodes.length > 0)
                     { 
                        if (node.childNodes.item(0).nodeName == "IMG")
                        {
                           node.childNodes.item(0).src = "minus.gif";
                        }
                     }
                  }

                  next_node.style.display = 'block';
               }
               // Collapse the branch if it IS visible
               else
               {
                  // Change the image (if there is an image)
                  if (node.childNodes.length > 0)
                  {
                     if (node.childNodes.length > 0)
                     { 
                        if (node.childNodes.item(0).nodeName == "IMG")
                        {
                           node.childNodes.item(0).src = "plus.gif";
                        }
                     }
                  }

                  next_node.style.display = 'none';
               }

            }
            </script>
            
            <style type="text/css">
               body {margin:0px;}
               pre {margin:0px;}

               ul.tree  li { list-style: none;  margin-left:10px;} 
               ul.tree  { margin:0px; padding:0px; margin-left:5px; font-size:0.95em; }
               ul.tree  li ul { margin-left:10px; padding:0px; }

               div#component {
                  background-color:white; 
                  border: 2px solid rgb(102,102,102); 
                  text-align:left;
                  margin-top: 1.5em;
                  padding: 0.7em;
               }

               div#extension {
                  background-color:#FDFDFD; 
                  border: 1px solid rgb(102,102,102); 
                  text-align:left;
                  margin-top: 1.0em;
                  padding: 0.7em;
               }

               div#logb {
                  text-align:left;
                  padding: 0.0em;
                  float: left;
                  background-color:#c0c0c0; 
                  border: double ; 
                  margin: 0.5em;
               }

               .bdotted {border-bottom: 1px dotted}
               .bdashed {border-bottom: 1px dashed}
               .bsolid {border-bottom: 1px solid}
               .bdouble {border-bottom: 1px double}
               .bgroove {border-bottom: 1px groove}
               .bridge {border-bottom: 1px ridge}
               .binset {border-bottom: 1px inset}
               .boutset {border-bottom: 1px outset}

               div#row1 {
                  background-color:#dfdfdf; 
               }
               div#row2 {
                  background-color:#f2f2f2; 
               }

               div#typedefs {
                  margin-left: 1.5em;
                  margin-top: 0.2em;
                  border: 1px dotted;
                  width: 52em;
               }

               div#tdn {
                  width: 10em;
               }

               .fullhr {
                  clear: both;
               }

               body {
                  text-align: center;
               }

               div#entire_page {
                  width:<xsl:value-of select="$main_width"/>;  
                  text-align: left;
                  margin-top: 0.4em;
                  margin-left: auto;
                  margin-right: auto;
               }
            </style>
            
         </head>
         <body bgcolor="{$background_color}">
            <a name="top" />
            <div id="entire_page">
            <table bgcolor="white" height="100%" bordercolor="{$background_color}" CELLSPACING="0" CELLPADDING="10">
               <tr height="100%">
                  <xsl:apply-templates select="document($main_menu)/doc/menu"/>

                  <!-- ************************************************************************* -->
                  <td  VALIGN="TOP" width="100%" style="border: 1px solid rgb(102,102,102);" >
                     <xsl:if test="title">
                        <center><h1> <xsl:value-of select="title" /> </h1></center>
                     </xsl:if>
                     <xsl:apply-templates select="body"/>
                  </td>
                  <!-- ************************************************************************* -->
                       <xsl:choose>
                           <xsl:when test="menu/@from_file">
                             <xsl:apply-templates select="document(menu/@from_file)/doc/menu">
                                 <xsl:with-param name="file_name" select="concat(substring-before(menu/@from_file,'.'),'.html')" />
                             </xsl:apply-templates>
                         </xsl:when>
                           <xsl:otherwise>
                          <xsl:apply-templates select="menu"/>
                           </xsl:otherwise>
                       </xsl:choose>         
                  
                  
                  <!-- ************************************************************************* -->
               </tr>
               
            </table>
                  
               <xsl:apply-templates select="components"/>
                  
            </div>
         </body>
      </html>
   </xsl:template>
   
         


   
   <!-- ************************************************************************* -->

   <xsl:template match="section">
      <xsl:param name="file_name" />
      <b><xsl:value-of select="name"/></b>
      <ul  class="tree">
         <xsl:for-each select="item">  
         <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/> 
            <xsl:apply-templates select=".">
               <xsl:with-param name="file_name" select="$file_name" />
            </xsl:apply-templates>
         </xsl:for-each>
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
   
   
   <xsl:template match="menu">
      <xsl:param name="file_name" />
      <td BGCOLOR="#F5F5F5" style="padding:7px; border: 1px solid rgb(102,102,102);" VALIGN="TOP" height="100%">
         <br/>
         <table WIDTH="{@width}" height="100%">
         <tr><td VALIGN="TOP">
         <xsl:apply-templates select="top">
            <xsl:with-param name="file_name" select="$file_name" />
         </xsl:apply-templates>
         </td><td width="1"></td></tr>
         <tr><td valign="bottom">
            <xsl:apply-templates select="bottom">
               <xsl:with-param name="file_name" select="$file_name" />
            </xsl:apply-templates>
         </td></tr>
         </table>
      </td>
   </xsl:template>
   
   <xsl:template match="item">
      <xsl:param name="file_name" />
  <li>
      <xsl:choose>
         <xsl:when test="@nolink = 'true'">
            <xsl:choose>
               <xsl:when test="name">
                  <a onclick="Toggle(this)" style="cursor: pointer;margin-left:-9px"><img src="plus.gif"/><font color="green"><u><xsl:value-of select="name"/></u></font></a>
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
                        <a href="{link}" style="float:right"><img src="right.gif" border="0"/></a>
                        <a onclick="Toggle(this)" style="cursor: pointer;margin-left:-9px" ><img src="plus.gif" border="0"/><font color="green"><u><xsl:value-of select="name"/></u></font></a>
      <xsl:apply-templates select="sub">
         <xsl:with-param name="file_name" select="$file_name" />
      </xsl:apply-templates> 
                     </xsl:when>
                     <xsl:otherwise>
                        <a href="{$file_name}#{name}" style="float:right"><img src="down.gif" border="0" /></a>
                        <a onclick="Toggle(this)" style="cursor: pointer;"><img src="plus.gif" border="0"/><font color="green"><u><xsl:value-of select="name"/></u></font></a>
      <xsl:apply-templates select="sub">
         <xsl:with-param name="file_name" select="$file_name" />
      </xsl:apply-templates>
                     </xsl:otherwise>
                  </xsl:choose>
               </xsl:when>
               <xsl:otherwise>
                  <xsl:choose>
                     <xsl:when test="link">
                        <a href="{link}"><xsl:value-of select="name"/></a>
                     </xsl:when>
                     <xsl:otherwise>
                        <a href="{$file_name}#{name}"><xsl:value-of select="name"/></a>
                     </xsl:otherwise>
                  </xsl:choose>
               </xsl:otherwise>
            </xsl:choose>
         </xsl:when>
         <xsl:otherwise>
            <a href="{$file_name}#{.}"><xsl:value-of select="."/></a>
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
      
   <xsl:template match="components">

      <xsl:for-each select="component">      
      <xsl:sort select="translate(name,$lcletters, $ucletters)"/> 
         <xsl:variable name="checked" select="@checked"/>

         <a name = "{name}">
         <div id="component"  >
      
         <a href="#top"><font size='2'><center>[top]</center></font></a>
         <h1 style="margin:0px;"><xsl:value-of select="name"/></h1>
         <BR/>
         <BR/>
         <xsl:apply-templates select="description"/>

         <xsl:if test="spec_file">
            <BR/>
            <xsl:choose>
               <xsl:when test="spec_file/@link = 'true'">
                  <BR/>
                  <b><font style='font-size:1.3em' color='#007700'>Specification: </font></b> <a href="{spec_file}.html#{name}"><xsl:value-of select="spec_file"/></a>
               </xsl:when>
               <xsl:otherwise>
                  <BR/>
                  <b><font style='font-size:1.3em' color='#007700'>Specification: </font></b> <a href="{spec_file}.html"><xsl:value-of select="spec_file"/></a>
               </xsl:otherwise>
            </xsl:choose>
         </xsl:if>
         <xsl:if test="file">
            <BR/><B>File to include: </B> <a href="{file}.html"><xsl:value-of select="file"/></a>
         </xsl:if>
         <xsl:if test="body_file">
            <BR/>
            The body for the <xsl:value-of select="name"/> component can be found 
            here: <a href="{body_file}.html#{name}"><xsl:value-of select="body_file"/></a>
         </xsl:if>
         <xsl:apply-templates select="examples"/>

         <xsl:apply-templates select="implementations">           
            <xsl:with-param name="checked" select="$checked" />
         </xsl:apply-templates>
               
         <xsl:choose>
            <xsl:when test="implementations"></xsl:when>
            <xsl:otherwise><br/><br/></xsl:otherwise>
         </xsl:choose>
      
         <xsl:if test="extensions">
            <br/>
            <center>
            <h1>Extensions to <xsl:value-of select="name"/></h1>
            </center>
            
            <xsl:for-each select="extensions/extension">
            <xsl:sort select="translate(name,$lcletters, $ucletters)"/> 
               <div id="extension">
               <a name="{name}"><B><font size='5'><xsl:value-of select="name"/></font></B></a><Br/>
               <BR/>
               <xsl:apply-templates select="description"/>
               <BR/>
               <BR/>
               <b><font style='font-size:1.3em' color='#007700'>Specification: </font></b> <a href="{spec_file}.html"><xsl:value-of select="spec_file"/></a>
               <xsl:apply-templates select="examples"/>
               <xsl:apply-templates select="implementations">           
                  <xsl:with-param name="checked" select="$checked" />
               </xsl:apply-templates>
               </div>
            </xsl:for-each>            
         </xsl:if>
      
      
            <!-- putting this empty center tag here, for whatever reason, prevents IE from
            messing up the space between these div blocks -->
            <center></center>
         </div>
         </a>
      </xsl:for-each>
   </xsl:template>      

   <xsl:template match="examples">
    <BR/><b>Code Examples: </b>
      <xsl:for-each select="example">
         <xsl:choose>
            <xsl:when test="position() = last()">
               <a href="{.}"><xsl:value-of select="position()"/></a>
            </xsl:when>
            <xsl:otherwise>
               <a href="{.}"><xsl:value-of select="position()"/></a>, 
            </xsl:otherwise>
         </xsl:choose>              
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
      
   
      <div id="typedefs"><table CELLSPACING='0' CELLPADDING='0' bgcolor="white" >       
         <xsl:for-each select="typedef">
         
            <xsl:choose>
               <xsl:when test="$checked = 'true'">
                  <tr><td bgcolor="{$gray}" valign="top"> 
                  <div id="tdn"><xsl:value-of select="name"/></div>  
                  </td><td width="100%" bgcolor="{$gray}"> 
                  <xsl:apply-templates select="description"/>
                  </td></tr>                 
               
                  <tr><td valign="top"> 
                  <div id="tdn"><xsl:value-of select="name"/>_c</div>
                  </td><td width="100%"> 
                  is a typedef for <xsl:value-of select="name"/> that checks its preconditions.             
                  </td></tr>                 
               </xsl:when>
               <xsl:otherwise>
                  <xsl:choose>
                     <xsl:when test="position() mod 2 = 0">
                        <tr><td valign="top"> 
                        <div id="tdn"><xsl:value-of select="name"/></div>  
                        </td><td width="100%"> 
                        <xsl:apply-templates select="description"/>
                        </td></tr>                                
                     </xsl:when>
                     <xsl:otherwise>
                        <tr><td bgcolor="{$gray}" valign="top"> 
                        <div id="tdn"><xsl:value-of select="name"/></div>  
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
   <xsl:template match="li">
      <li>
         <xsl:apply-templates/>
       </li>
   </xsl:template>   
   <xsl:template match="ul">
      <ul>
         <xsl:apply-templates/>
       </ul>
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
      <font color="{@color}">
         <xsl:apply-templates/>
       </font>
   </xsl:template>   
   <xsl:template match="img">
      <img src="{@src}" border="0" height="{@height}" width="{@width}" alt="{@alt}">
         <xsl:apply-templates/>
       </img>
   </xsl:template>   

   <xsl:template name="term_list_go">
      <xsl:param name="num"/>
      <xsl:if test="$num &lt; 27">
            <ul>
               <xsl:variable name="cur_letter" select="substring($ucletters, $num, 1)"/>

               <a name="{$cur_letter}"/>
               <h1><xsl:value-of select="$cur_letter"/></h1>
               <xsl:for-each select="term">
               <xsl:sort order="ascending" select="translate(@name,$lcletters, $ucletters)"/>
               <xsl:if test="$cur_letter = substring(translate(@name,$lcletters, $ucletters),1,1)">
               <xsl:choose>
                  <xsl:when test="@link">
                     <li><a href="{@link}"><xsl:value-of select="@name"/></a></li>
                  </xsl:when>
                  <xsl:when test="@file">
                     <li><a href="{@file}#{@name}"><xsl:value-of select="@name"/></a></li>
                  </xsl:when>
                  <xsl:otherwise>
                     <li> <xsl:value-of select="@name"/>
                        <ul>
                        <xsl:for-each select="term">
                        <xsl:sort order="ascending" select="translate(@name,$lcletters, $ucletters)"/> 
                           <li><a href="{@link}"><xsl:value-of select="@name"/></a></li>
                        </xsl:for-each>
                        </ul>
                     </li>
                  </xsl:otherwise>
               </xsl:choose>
               </xsl:if>
               </xsl:for-each>
            </ul>

      <xsl:call-template name="term_list_go" >
         <xsl:with-param name="num" select="$num + 1"/>
      </xsl:call-template>

      </xsl:if>
   </xsl:template>   


   <xsl:template match="term_list">
      <center>
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
      </center>
      <xsl:call-template name="term_list_go" >
         <xsl:with-param name="num" select="1"/>
      </xsl:call-template>
   </xsl:template>   



   <!--  This function turns a string of the form 2006-03-21T02:35:20.510956Z into a nice 
   normal looking date -->
   <xsl:template name="format-date">
      <xsl:param name="xsd-date"/>
      <xsl:variable name="date" select="substring-before($xsd-date,'T')"/>
      <xsl:variable name="time" select="substring-before(substring-after($xsd-date,'T'),'.')"/>

      <xsl:variable name="year" select="substring($date,1,4)"/>
      <xsl:variable name="month" select="substring($date,6,2)"/>
      <xsl:variable name="day" select="substring($date,9,2)"/>
      <xsl:variable name="hour" select="substring($time,1,2)"/>
      <xsl:variable name="minute" select="substring($time,4,2)"/>
      <xsl:variable name="second" select="substring($time,7,2)"/>

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

      <xsl:value-of select="$day"/>, <xsl:value-of select="$year"/>
      (<xsl:value-of select="$hour"/>:<xsl:value-of select="$minute"/>:<xsl:value-of select="$second"/> UTC)

   </xsl:template>



   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->
   <!-- *******    Subversion stylesheet stfuff   ******************** -->
   <!-- ************************************************************************* -->
   <!-- ************************************************************************* -->


   
   <!-- ************************************************************************* -->
   
   
   <xsl:template match="log">
      <xsl:for-each select="logentry">
      <xsl:sort order="descending" data-type="number" select="./@revision"/>
      <u>Revision</u>: <xsl:value-of select="@revision"/> <br/>
      <u>Date</u>: <xsl:call-template name="format-date"><xsl:with-param name="xsd-date" select="date"/></xsl:call-template> <br/>
            <xsl:apply-templates select="msg"/>
            <xsl:apply-templates select="paths"/>
      <hr class="fullhr"/>
      </xsl:for-each>
   </xsl:template>
   
         


   
   <!-- ************************************************************************* -->

   <xsl:template name="paths">
    <xsl:param name="type"/>
    <xsl:param name="name"/>
    <xsl:param name="color"/>

     <xsl:if test="path[@action=$type]">

      <div id="logb">
         <div class="bsolid"><b><font color="{$color}"><xsl:value-of select="$name"/></font></b></div>
            <xsl:for-each select="path[@action = $type]">
            <xsl:sort select="."/>
               <xsl:choose>
                  <xsl:when test="position() mod 2 = 0">
                     <div id="row1"><xsl:value-of select="."/></div>
                  </xsl:when>
                  <xsl:otherwise>
                     <div id="row2"><xsl:value-of select="."/></div>
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
   </xsl:template>

   <xsl:template match="msg">
    <pre><xsl:value-of select="."/></pre>
   </xsl:template>

   
   <!-- ************************************************************************* -->






   
</xsl:stylesheet>
