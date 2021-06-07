/////////////////////////////////////////////////////////////////////////////
// Name:        wx/generic/gridsel.h
// Purpose:     wxGridSelection
// Author:      Stefan Neis
// Modified by:
// Created:     20/02/2000
// Copyright:   (c) Stefan Neis
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_GENERIC_GRIDSEL_H_
#define _WX_GENERIC_GRIDSEL_H_

#include "wx/defs.h"

#if wxUSE_GRID

#include "wx/grid.h"

#include "wx/vector.h"

typedef wxVector<wxGridBlockCoords> wxVectorGridBlockCoords;

// Note: for all eventType arguments of the methods of this class wxEVT_NULL
//       may be passed to forbid events generation completely.
class WXDLLIMPEXP_CORE wxGridSelection
{
public:
    wxGridSelection(wxGrid *grid,
                    wxGrid::wxGridSelectionModes sel = wxGrid::wxGridSelectCells);

    bool IsSelection();
    bool IsInSelection(int row, int col) const;
    bool IsInSelection(const wxGridCellCoords& coords) const
    {
        return IsInSelection(coords.GetRow(), coords.GetCol());
    }

    void SetSelectionMode(wxGrid::wxGridSelectionModes selmode);
    wxGrid::wxGridSelectionModes GetSelectionMode() { return m_selectionMode; }
    void SelectRow(int row, const wxKeyboardState& kbd = wxKeyboardState());
    void SelectCol(int col, const wxKeyboardState& kbd = wxKeyboardState());
    void SelectBlock(int topRow, int leftCol,
                     int bottomRow, int rightCol,
                     const wxKeyboardState& kbd = wxKeyboardState(),
                     wxEventType eventType = wxEVT_GRID_RANGE_SELECTED);
    void SelectBlock(const wxGridCellCoords& topLeft,
                     const wxGridCellCoords& bottomRight,
                     const wxKeyboardState& kbd = wxKeyboardState(),
                     wxEventType eventType = wxEVT_GRID_RANGE_SELECTED)
    {
        SelectBlock(topLeft.GetRow(), topLeft.GetCol(),
                    bottomRight.GetRow(), bottomRight.GetCol(),
                    kbd, eventType);
    }

    // This function replaces all the existing selected blocks (which become
    // redundant) with a single block covering the entire grid.
    void SelectAll();

    void DeselectBlock(const wxGridBlockCoords& block,
                       const wxKeyboardState& kbd = wxKeyboardState(),
                       wxEventType eventType = wxEVT_GRID_RANGE_SELECTED);

    // Note that this method refreshes the previously selected blocks and sends
    // an event about the selection change.
    void ClearSelection();

    void UpdateRows( size_t pos, int numRows );
    void UpdateCols( size_t pos, int numCols );

    // Extend (or shrink) the current selection block (creating it if
    // necessary, i.e. if there is no selection at all currently or if the
    // current current cell isn't selected, as in this case a new block
    // containing it is always added) to the one specified by the start and end
    // coordinates of its opposite corners (which don't have to be in
    // top/bottom left/right order).
    //
    // Note that blockStart is equal to wxGrid::m_currentCellCoords almost
    // always, but not always (the exception is when we scrolled out from
    // the top of the grid and select a column or scrolled right and select
    // a row: in this case the lowest visible row/column will be set as
    // current, not the first one).
    //
    // Both components of both blockStart and blockEnd must be valid.
    //
    // This function sends an event notifying about the selection change using
    // the provided event type, which is wxEVT_GRID_RANGE_SELECTED by default,
    // but may also be wxEVT_GRID_RANGE_SELECTING, when the selection is not
    // final yet.
    //
    // Return true if the current block was actually changed.
    bool ExtendCurrentBlock(const wxGridCellCoords& blockStart,
                            const wxGridCellCoords& blockEnd,
                            const wxKeyboardState& kbd,
                            wxEventType eventType = wxEVT_GRID_RANGE_SELECTED);


    // Return the coordinates of the cell from which the selection should
    // continue to be extended. This is normally the opposite corner of the
    // last selected block from the current cell coordinates.
    //
    // If there is no selection, just returns the current cell coordinates.
    wxGridCellCoords GetExtensionAnchor() const;

    wxGridCellCoordsArray GetCellSelection() const;
    wxGridCellCoordsArray GetBlockSelectionTopLeft() const;
    wxGridCellCoordsArray GetBlockSelectionBottomRight() const;
    wxArrayInt GetRowSelection() const;
    wxArrayInt GetColSelection() const;

    wxVectorGridBlockCoords& GetBlocks() { return m_selection; }

    void EndSelecting();

private:
    void SelectBlockNoEvent(const wxGridBlockCoords& block)
    {
        SelectBlock(block.GetTopRow(), block.GetLeftCol(),
                    block.GetBottomRow(), block.GetRightCol(),
                    wxKeyboardState(), false);
    }

    // Really select the block and don't check for the current selection mode.
    void Select(const wxGridBlockCoords& block,
                const wxKeyboardState& kbd,
                wxEventType eventType);

    // Ensure that the new "block" becomes part of "blocks", adding it to them
    // if necessary and, if we do it, also removing any existing elements of
    // "blocks" that become unnecessary because they're entirely contained in
    // the new "block". However note that we may also not to have to add it at
    // all, if it's already contained in one of the existing blocks.
    //
    // We don't currently check if the new block is contained by several
    // existing blocks, as this would be more difficult and doesn't seem to be
    // really needed in practice.
    void MergeOrAddBlock(wxVectorGridBlockCoords& blocks,
                         const wxGridBlockCoords& block);

    // All currently selected blocks. We expect there to be a relatively small
    // amount of them, even for very large grids, as each block must be
    // selected by the user, so we store them unsorted.
    //
    // Selection may be empty, but if it isn't, the last block is special, as
    // it is the current block, which is affected by operations such as
    // extending the current selection from keyboard.
    wxVectorGridBlockCoords             m_selection;

    wxGrid                              *m_grid;
    wxGrid::wxGridSelectionModes        m_selectionMode;

    wxDECLARE_NO_COPY_CLASS(wxGridSelection);
};

#endif  // wxUSE_GRID
#endif  // _WX_GENERIC_GRIDSEL_H_
