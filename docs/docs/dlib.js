
function init_page()
{
   if (navigator.appVersion.indexOf("Win")!=-1)
   {
      var a = document.getElementById("download_button");
      a.href = a.href.replace("tar.bz2", "zip");
   }
}
window.onload = init_page;

// --------------------------------------------------------------
//  Tree collapse stuff
// --------------------------------------------------------------

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
function BigToggle(node)
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
               node.childNodes.item(0).src = "bigminus.gif";
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
               node.childNodes.item(0).src = "bigplus.gif";
            }
         }
      }

      next_node.style.display = 'none';
   }

}

