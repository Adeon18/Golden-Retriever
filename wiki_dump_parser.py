# MODIFIED copy of https://github.com/Grasia/wiki-scripts/blob/master/wiki_dump_parser/wiki_dump_parser.py
from pathlib import Path
import xml.parsers.expat
import sys
import csv

__version__ = '2.0.2'

Debug = False

csv_separator = ","

def xml_to_csv(filename):
    ### BEGIN xml_to_csv var declarations ###
    # Shared variables for parser subfunctions:
    ## output_csv, _current_tag, _parent
    ## page_id, page_title, page_ns, revision_id, timestamp, contributor_id, contributor_name, bytes_var
    filename = Path(filename)

    output_csv = None
    _parent = None
    _current_tag = ''
    page_id = page_title = page_ns = revision_text = revision_id = timestamp = contributor_id = contributor_name = bytes_var = ''

    def start_tag(tag, attrs):
        nonlocal output_csv, _current_tag, _parent
        nonlocal bytes_var

        _current_tag = tag

        if tag == 'text':
            if 'bytes' in attrs:
                bytes_var = attrs['bytes']
            else: # There's a 'deleted' flag or no info about bytes of the edition
                bytes_var = '-1'
        elif tag == 'page' or tag == 'revision' or tag == 'contributor':
            _parent = tag

        if tag == 'upload':
            print("!! Warning: '<upload>' element not being handled", file=sys.stderr)

    def data_handler(data):
        nonlocal output_csv, _current_tag, _parent
        nonlocal page_id,page_title,page_ns,revision_text,revision_id,timestamp,contributor_id,contributor_name

        if _current_tag == '': # Don't process blank "orphan" data between tags!!
            return

        if _parent:
            if _parent == 'page':
                if _current_tag == 'title':
                    page_title = data
                elif _current_tag == 'id':
                    page_id = data
                    if Debug:
                        print("Parsing page " + page_id )
                elif _current_tag == 'ns':
                    page_ns = data
            elif _parent == 'revision':
                if _current_tag == 'id':
                    revision_id = data
                elif _current_tag == 'timestamp':
                    timestamp = data
                elif _current_tag == 'text':
                    revision_text = data
            elif _parent == 'contributor':
                if _current_tag == 'id':
                    contributor_id = data
                elif _current_tag == 'username':
                    contributor_name = data
                elif _current_tag == 'ip':
                    contributor_id = data
                    contributor_name = 'Anonymous'

    def end_tag(tag):
        nonlocal output_csv, _current_tag, _parent
        nonlocal page_id,page_title,page_ns,revision_text,revision_id,timestamp,contributor_id,contributor_name,bytes_var


        def has_empty_field(l):
            field_empty = False
            i = 0
            while not field_empty and i<len(l):
                field_empty = (l[i] == '')
                i = i + 1
            return field_empty


        # uploading one level of parent if any of these tags close
        if tag == 'page':
            _parent = None
        elif tag == 'revision':
            _parent = 'page'
        elif tag == 'contributor':
            _parent = 'revision'

        # print revision to revision output csv
        if tag == 'revision':

            revision_row = [page_id, page_title, page_ns,
                            revision_id, timestamp,
                            contributor_id,contributor_name,
                            bytes_var, revision_text]

            # Do not print (skip) revisions that has any of the fields not available
            if not has_empty_field(revision_row):
                output_csv.writerow(revision_row)
            else:
                print("The following line has incomplete info and therefore it's been removed from the dataset:")
                print(revision_row)

            # Debug lines to standard output
            if Debug:
                print(csv_separator.join(revision_row))

            # Clearing data that has to be recalculated for every row:
            revision_id = timestamp = contributor_id = contributor_name = bytes_var = ''

        _current_tag = '' # Very important!!! Otherwise blank "orphan" data between tags remain in _current_tag and trigger data_handler!! >:(


    ### BEGIN xml_to_csv body ###

    # Initializing xml parser
    with open(filename, 'rb') as input_file, open(filename.with_suffix('.csv'), 'w', encoding='utf8') as output_csv_f:
        output_csv = output_csv_f
        parser = xml.parsers.expat.ParserCreate()

        parser.StartElementHandler = start_tag
        parser.EndElementHandler = end_tag
        parser.CharacterDataHandler = data_handler
        parser.buffer_text = True
        parser.buffer_size = 32 * 1024 * 1024

        # writing header in the output CSV file
        output_csv = csv.writer(output_csv)
        output_csv.writerow(["page_id","page_title","page_ns","revision_id","timestamp","contributor_id","contributor_name","bytes","text"])

        # Parsing XML and writing processed data to output CSV
        print("Processing...")
        parser.ParseFile(input_file)
        print("Done processing")

    return True