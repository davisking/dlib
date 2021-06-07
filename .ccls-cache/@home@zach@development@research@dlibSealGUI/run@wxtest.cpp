#include <wx/wx.h>

class Simple : public wxFrame
{
public:
    Simple(const wxString& title)
		: wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(250, 150))
	{
		Centre();
	}
};

class MyApp : public wxApp
{
public:
	bool OnInit()
	{
		Simple *simple = new Simple(wxT("Simple"));
		simple->Show(true);
		return true;
	}
};

wxIMPLEMENT_APP(MyApp);
