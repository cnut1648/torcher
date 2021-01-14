from torcherist.algorithms.nlp import clean_html, lowercase_strip_punc

class CorpusCreator(object):
    """
        A base class for objects that convert common schema
        job listings into a corpus in documnet level suitable for use by
        machine learning algorithms or specific tasks.

    Example:
    ```python
    from skills_ml.job_postings.common_schema import JobPostingCollectionSample
    from skills_ml.job_postings.corpora.basic import CorpusCreator

    job_postings_generator = JobPostingCollectionSample()

    # Default will include all the cleaned job postings
    corpus = CorpusCreator(job_postings_generator)

    # For getting a the raw job postings without any cleaning
    corpus = CorpusCreator(job_postings_generator, raw=True)
    ```


    Attributes:
        job_posting_generator (generator):  an iterable that generates JSON strings.
                                Each string is expected to represent a job listing
                                conforming to the common schema
                                See sample_job_listing.json for an example of this schema
        document_schema_fields (list): an list of schema fields to be included
        raw (bool): a flag whether to return the raw documents or transformed documents

    Yield:
        (dict): a dictinary only with selected fields as keys and corresponding raw/cleaned value
    """
    def __init__(self, job_posting_generator=None, document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills'],
                 raw=False):
        self.job_posting_generator = job_posting_generator
        self.raw = raw
        self.document_schema_fields = document_schema_fields
        self.join_spaces = ' '.join
        self.key = ['onet_soc_code']


    @property
    def metadata(self):
        meta_dict = {'corpus_creator': ".".join([self.__module__ , self.__class__.__name__])}
        if self.job_posting_generator:
            meta_dict.update(self.job_posting_generator.metadata)
        return meta_dict

    def _clean(self, document):
        for f in self.document_schema_fields:
            try:
                cleaned = clean_html(document[f]).replace('\n','')
                cleaned = " ".join(cleaned.split())
                document[f] = cleaned
            except KeyError:
                pass
        return document

    def _transform(self, document):
        if self.raw:
            return self._join(document)
        else:
            return self._clean(document)

    def _join(self, document):
        return self.join_spaces([
            document.get(field, '') for field in self.document_schema_fields
        ])

    def __iter__(self):
        for document in self.job_posting_generator:
            document = {key: document[key] for key in self.document_schema_fields}
            yield self._transform(document)


class SimpleCorpusCreator(CorpusCreator):
    """
        An object that transforms job listing documents by picking
        important schema fields and returns them as one large lowercased string
    """
    def _clean(self, document):
        return self.join_spaces([
            lowercase_strip_punc(document.get(field, ''))
            for field in self.document_schema_fields
        ])
