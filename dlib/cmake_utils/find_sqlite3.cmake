set(SQLite3_FOUND False)

find_library(sqlite sqlite3)
find_path(sqlite_path sqlite3.h)
    
if (sqlite AND sqlite_path)
    mark_as_advanced(sqlite sqlite_path)
    set(SQLite3_FOUND True)
    set(SQLite3_LIBRARIES ${sqlite})
    set(SQLite3_INCLUDE_DIRS ${sqlite_path})
endif()
    
