# slepemapy.cz

The data set is based on the online system http://slepemapy.cz. The system is
available in Czech, English, and Spanish, most users are from Czech Republic
(78 %) and Slovakia (10 %). The system uses adaptive algorithms for choosing
questions, these algorithms are described in detail in [1] and  [2]. This is
the first publicly available version of the data set, it is static and captures
student interactions up to 21 May 2015. The basic statistics of the data set
are as follows:

  - 91,331 students;
  - 1,459 geographical items
  - 10,087,306 answers

## Description

The dataset contains 3 CSV files (semicolons are used as delimiter):

 - with the answers of users practicing location of places;
 - places;
 - types of places.

### Answers

|        Column       | Description                                                                                          |
|:-------------------:|:----------------------------------------------------------------------------------------------------:|
|          id         | answer identifier                                                                                    |
|         user        | user's identifier                                                                                    |
|     place_asked     | identifier of the asked place                                                                        |
|    place_answered   | identifier of the answered place, empty if the user answered "I don't know"                          |
|         type        | type of the answer: (1) find the given place on the map; (2) pick the name for the highlighted place |
|        options      | list of identifiers of options (the asked place included)                                            |
|       inserted      | datetime (yyyy-mm-dd HH:mm:ss) when the answer was inserted to the system                            |
|     response_time   | how much time the answer took (measured in milliseconds)                                             |
|       place_map     | identifier of the place representing a map for which the question was asked                          |
|      ip_country     | country retrieved from the user’s IP address                                                         |
|        ip_id        | meaningless identifier of the user’s IP address                                                      |

### Places

| Column | Description                                               |
|:------:|:----------------------------------------------------------|
|   id   | identifier of the place                                   |
|  code  | code of the place (ISO 3166-1 alpha-2 if it is possible)  |
|  name  | name of the place                                         |
|  type  | type of the place (described in a seperate file)          |


### Place Types

| Column | Description            |
|:------:|:-----------------------|
|   id   | identifier of the type |
|  name  | name of the type       |


## Ethical and privacy considerations:

The used educational system is used mainly by students in schools or by
students preparing for exams. Nevertheless, it is an open online system which
can be used by anybody and details about individual users are not available.
Users are identified only by their anonymous ID. Users can log into the system
using their Google or Facebook accounts; but this login is used only for
identifying the user within the system, it is not included in the data set.
Unlogged users are tracked using web browser cookies. The system also logs IP
address from which users access the system, the IP address is included in the
data set in anonymized form. We separately encode the country of origin, which
can be useful for analysis and its inclusion is not a privacy concern. The rest
of the IP address is replaced by meaningless identifier to preserve privacy.

## Terms of Use

The data set is available at http://www.fi.muni.cz/adaptivelearning/data/slepemapy/

### License

This data set is made available under Open Database License whose full text can
be found at http://opendatacommons.org/licenses/odbl/. Any rights in individual
contents of the database are licensed under the Database Contents License whose
text can be found http://opendatacommons.org/licenses/dbcl/

### Citation

Please cite the following paper when you use our data set:

```
@article{
  author={Papou{\v{s}}ek, Jan and Pel{\'a}nek, Radek and Stanislav, V{\'\i}t},
  title={Adaptive Geography Practice Data Set},
  journal={Journal of Learning Analytics},
  year={2015},
  issn={1929-7750}
}
```

## References

  - **[1]** Papoušek, J., Pelánek, R. & Stanislav, V. Adaptive Practice of Facts in Domains with Varied Prior Knowledge. In Educational Data Mining, 2014.
  - **[2]** Papoušek, J., & Pelánek, R. Impact of adaptive educational system behaviour on student motivation. In Artificial Intelligence in Education, 2015.
