<?xml version="1.0" encoding="ISO-8859-1" ?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
	<xsl:output method='html' version='1.0' encoding='UTF-8' indent='yes' />
	
	
	<xsl:variable name="main_menu" select="/htmlhelp_toc/main_menu_file"/>
   <xsl:variable name="docs" select="/htmlhelp_toc/docs_folder"/>
   <xsl:variable name="show_include_file" select="/htmlhelp_toc/show_include_file"/>

	<!-- ************************************************************************* -->
   <xsl:variable name="lcletters">abcdefghijklmnopqrstuvwxyz </xsl:variable>
   <xsl:variable name="ucletters">ABCDEFGHIJKLMNOPQRSTUVWXYZ </xsl:variable>
	<!-- ************************************************************************* -->
	
	
	<xsl:template match="/htmlhelp_toc">
      <HTML>
      <HEAD>
      </HEAD><BODY>
      <UL>


                  <xsl:apply-templates select="document($main_menu)/doc/menu"/>







      </UL>
      </BODY></HTML>
	</xsl:template>
	
			


	
	<!-- ************************************************************************* -->


	<xsl:template match="section">
		<xsl:param name="html_file" />
		<xsl:param name="xml_file" />

       <xsl:choose> 
         <xsl:when test="count(/doc/menu/top/section) != 1">
            <LI> <OBJECT type="text/sitemap">
               <param name="Name" value="{name}"/>
               </OBJECT>
            </LI>
            <UL>
               <xsl:for-each select="item | chm/item">  
               <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/> 
                  <xsl:apply-templates select=".">					
                     <xsl:with-param name="xml_file" select="$xml_file"/>
                     <xsl:with-param name="html_file" select="$html_file"/>
                  </xsl:apply-templates>
               </xsl:for-each>
            </UL>
         </xsl:when>
         <xsl:otherwise>
            <xsl:for-each select="item | chm/item">  
            <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/> 
               <xsl:apply-templates select=".">					
                  <xsl:with-param name="xml_file" select="$xml_file"/>
                  <xsl:with-param name="html_file" select="$html_file"/>
               </xsl:apply-templates>
            </xsl:for-each>
         </xsl:otherwise>
      </xsl:choose>
	</xsl:template>
	
	<!-- ************************************************************************* -->

	<xsl:template match="menu">
		<xsl:param name="html_file" />
		<xsl:param name="xml_file" />
      <xsl:for-each select="top/section">  
         <xsl:apply-templates select=".">					
            <xsl:with-param name="xml_file" select="$xml_file"/>
            <xsl:with-param name="html_file" select="$html_file"/>
         </xsl:apply-templates>
      </xsl:for-each>
	</xsl:template>
	
	<!-- ************************************************************************* -->
	<xsl:template match="item">
		<xsl:param name="html_file" />
		<xsl:param name="xml_file" />
    <xsl:choose>
      <xsl:when test="name != ''">
         <LI><OBJECT type="text/sitemap">
            <param name="Name" value="{name}"/>
               <xsl:choose>
                  <xsl:when test="link">
                     <param name="Local" value="{$docs}\{link}"/>
                  </xsl:when>
                  <xsl:when test="@nolink = 'true'">
                  </xsl:when>
                  <xsl:otherwise>
                     <param name="Local" value="{$docs}\{$html_file}#{name}"/>
                  </xsl:otherwise>
               </xsl:choose>

            </OBJECT>
         </LI>

         <xsl:choose>
            <xsl:when test="sub">
               <UL>
                  <xsl:for-each select="sub/item">  
                  <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/> 
                     <xsl:apply-templates select=".">					
                        <xsl:with-param name="xml_file" select="$xml_file"/>
                        <xsl:with-param name="html_file" select="$html_file"/>
                     </xsl:apply-templates>
                  </xsl:for-each>
               </UL>
            </xsl:when>
            <xsl:when test="chm_sub">
               <UL>
						<xsl:apply-templates select="document(chm_sub)/doc/menu">					
							<xsl:with-param name="xml_file" select="chm_sub"/>
							<xsl:with-param name="html_file" select="link"/>
						</xsl:apply-templates>
               </UL>
            </xsl:when>
         </xsl:choose>
      </xsl:when>
      <xsl:when test="@nolink = 'true'">
      </xsl:when>
      <xsl:otherwise>
         <LI><OBJECT type="text/sitemap">
            <param name="Name" value="{.}"/>
               <param name="Local" value="{$docs}\{$html_file}#{.}"/>
            </OBJECT>
         </LI>
         <xsl:variable name="cname" select="."/>
         <xsl:for-each select="document($xml_file)/doc/components/component">
            <xsl:if test="name = $cname">
               <UL>
                  <xsl:if test="spec_file">
                     <xsl:choose>
                        <xsl:when test="spec_file/@link = 'true'">
                           <LI> <OBJECT type="text/sitemap">
                              <param name="Name" value="specification"/>
                                 <param name="Local" value="{$docs}\{spec_file}.html#{name}"/>
                              </OBJECT>
                           </LI>
                        </xsl:when>
                        <xsl:otherwise>
                           <LI> <OBJECT type="text/sitemap">
                              <param name="Name" value="specification"/>
                                 <param name="Local" value="{$docs}\{spec_file}.html"/>
                              </OBJECT>
                           </LI>
                        </xsl:otherwise>
                     </xsl:choose>
                  </xsl:if>
                  <xsl:if test="$show_include_file = 'true'">
                     <xsl:if test="file">
                        <xsl:choose>
                           <xsl:when test="spec_file/@link = 'true' "> 
                              <LI> <OBJECT type="text/sitemap">
                                    <param name="Name" value="include file"/>
                                    <param name="Local" value="{$docs}\{file}.html"/>
                                 </OBJECT>
                              </LI>
                           </xsl:when>
                           <xsl:when test="spec_file != file">
                              <LI> <OBJECT type="text/sitemap">
                                    <param name="Name" value="include file"/>
                                    <param name="Local" value="{$docs}\{file}.html"/>
                                 </OBJECT>
                              </LI>
                           </xsl:when>
                        </xsl:choose>
                     </xsl:if>
                  </xsl:if>
                  <xsl:if test="body_file">
                     <LI> <OBJECT type="text/sitemap">
                        <param name="Name" value="body"/>
                           <param name="Local" value="{$docs}\{body_file}.html#{name}"/>
                        </OBJECT>
                     </LI>
                  </xsl:if>
                  <xsl:if test="extensions">
                     <LI> <OBJECT type="text/sitemap">
                        <param name="Name" value="extensions"/>
                        </OBJECT>
                     </LI>
                     <UL>
                        <xsl:for-each select="extensions/extension">
                        <LI> <OBJECT type="text/sitemap">
                           <param name="Name" value="{name}"/>
                              <param name="Local" value="{$docs}\{spec_file}.html"/>
                           </OBJECT>
                        </LI>
                        </xsl:for-each>
                     </UL>
                  </xsl:if>
               </UL>
            </xsl:if>
         </xsl:for-each>
      </xsl:otherwise>
    </xsl:choose>
	</xsl:template>	
	
	<!-- ************************************************************************* -->
	<xsl:template match="sub">
		<ul>
			<xsl:for-each select="item">  
         <xsl:sort select="translate(concat(name,.),$lcletters, $ucletters)"/> 
				<xsl:apply-templates select="."/>
			</xsl:for-each>
		</ul>
	</xsl:template>	
	
	<!-- ************************************************************************* -->
		
	
</xsl:stylesheet>
