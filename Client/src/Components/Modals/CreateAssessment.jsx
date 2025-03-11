import { useState, useEffect } from "react";
import axios from "axios";
import toast from "react-hot-toast";
import PropTypes from "prop-types";
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  Select,
  SelectItem,
} from "@nextui-org/react";

const CreateSuccess = ({ show, onHide }) => {
  const handleSuccessClick = () => {
    onHide();
    window.location.reload(); // Consider updating state instead of reloading.
  };

  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="contained-modal-title-vcenter"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader id="contained-modal-title-vcenter">
          Activity Created Successfully!
        </ModalHeader>
        <ModalBody>
          <p>You have created an activity for your students!</p>
        </ModalBody>
        <ModalFooter>
          <Button color="primary" onClick={handleSuccessClick}>
            Close
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

const CreateAssessment = ({ show, handleClose, userId, section }) => {
  const [modalShow, setModalShow] = useState(false);
  const [words, setWords] = useState([]);
  const [sentences, setSentences] = useState([]);
  const [filteredWords, setFilteredWords] = useState([]);
  const [assessmentExist, setAssessmentExist] = useState({
    "Assessment 1": false,
    "Assessment 2": false,
    "Assessment 3": false,
    "Assessment 4": false,
  });
  const [data, setData] = useState({
    Period: "",
    Type: "",
    Title: "",
    Item1: "",
    Item2: "",
    Item3: "",
    Item4: "",
    Item5: "",
    Item6: "",
    Item7: "",
    Item8: "",
    Item9: "",
    Item10: "",
  });

  const [errors, setErrors] = useState({});

  useEffect(() => {
    fetchAssessments();
    fetchImportWord();
    fetchImportSentence();
  }, []);

  const fetchAssessments = () => {
    axios
      .get(`/api/getAssessments`)
      .then((response) => {
        const assessments = response.data.filter(
          (activity) => activity.UserID === userId && activity.Section === section
        );
        const assessmentStatus = {
          "Assessment 1": assessments.some((activity) => activity.Type === "Pagbabaybay"),
          "Assessment 2": assessments.some((activity) => activity.Type === "Pantig"),
          "Assessment 3": assessments.some((activity) => activity.Type === "Salita"),
          "Assessment 4": assessments.some((activity) => activity.Type === "Pagbabasa"),
        };
        setAssessmentExist(assessmentStatus);
      })
      .catch((err) => console.log(err));
  };

  const fetchImportWord = () => {
    axios
      .get("/api/getImportWord")
      .then((response) => {
        setWords(response.data);
      })
      .catch((err) => console.log(err));
  };

  const fetchImportSentence = () => {
    axios
      .get("/api/getSentence")
      .then((response) => {
        setSentences(response.data);
      })
      .catch((err) => console.log(err));
  };

  useEffect(() => {
    if (data.Type && data.Type !== "Pagbabasa") {
      const filtered = words.filter((word) => word.Type === data.Type);
      setFilteredWords(filtered);
    } else {
      setFilteredWords([]);
    }
  }, [data.Type, words]);

  // Handle selection change for word items
  const handleWordChange = (itemKey, selectedWord) => {
    const selectedItem = filteredWords.find(
      (word) => word.Word === selectedWord
    );
    setData((prevData) => ({
      ...prevData,
      [itemKey]: selectedItem ? selectedItem.ItemCode : "",
    }));
  };

  const validateInputs = () => {
    let isValid = true;
    let newErrors = {};

    if (!data.Period) {
      newErrors.Period = "Grading Period is required.";
      isValid = false;
    }
    if (!data.Type) {
      newErrors.Type = "Type of Assessment is required.";
      isValid = false;
    }

    if (data.Type === "Pagbabasa" && !data.Title) {
      newErrors.Title = "Title is required for Pagbabasa.";
      isValid = false;
    }

    if (data.Type !== "Pagbabasa") {
      for (let i = 1; i <= 10; i++) {
        if (!data[`Item${i}`]) {
          newErrors[`Item${i}`] = `Item ${i} is required.`;
          isValid = false;
        }
      }
    }

    setErrors(newErrors);
    return isValid;
  };

  const createAct = async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      toast.error("Please fill out all required fields.");
      return;
    }

    try {
      const response = await axios.post(`/api/submitAssessment`, {
        UserID: userId,
        Section: section,
        Period: data.Period,
        Type: data.Type,
        Title: data.Title,
        Item1: data.Item1,
        Item2: data.Item2,
        Item3: data.Item3,
        Item4: data.Item4,
        Item5: data.Item5,
        Item6: data.Item6,
        Item7: data.Item7,
        Item8: data.Item8,
        Item9: data.Item9,
        Item10: data.Item10,
      });
      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        setData({
          Period: "",
          Type: "",
          Title: "",
          Item1: "",
          Item2: "",
          Item3: "",
          Item4: "",
          Item5: "",
          Item6: "",
          Item7: "",
          Item8: "",
          Item9: "",
          Item10: "",
        });
        toast.success("Created Activity Successfully.");
        setModalShow(true);
        handleClose();
      }
    } catch (error) {
      console.log(error);
      toast.error("Failed to create the activity. Please try again.");
    }
  };

  return (
    <>
      <Modal
        isOpen={show}
        onClose={handleClose}
        size="lg"
        aria-labelledby="contained-modal-title-vcenter"
        isDismissable={false}
        isKeyboardDismissDisabled={true}
        placement="center"
        scrollBehavior="inside"
      >
        <ModalContent className="w-full md:w-[50vw] max-w-full max-h-[80vh] overflow-y-auto bg-white p-4 rounded-lg">
          <ModalHeader
            id="contained-modal-title-vcenter"
            className="text-lg font-bold"
          >
            Create New Activity
          </ModalHeader>
          <form onSubmit={createAct} className="space-y-4">
            <ModalBody>
              <Select
                labelPlacement="outside"
                label="Grading Period"
                placeholder="Select Grading Period"
                value={data.Period}
                onChange={(e) => setData({ ...data, Period: e.target.value })}
                className="w-full my-2"
                isInvalid={!!errors.Period}
                errorMessage={errors.Period}
              >
                <SelectItem key="1">Grading Period 1</SelectItem>
                <SelectItem key="2">Grading Period 2</SelectItem>
                <SelectItem key="3">Grading Period 3</SelectItem>
                <SelectItem key="4">Grading Period 4</SelectItem>
              </Select>

              <Select
                labelPlacement="outside"
                label="Type of Assessment"
                placeholder="Select Type of Assessment"
                value={data.Type}
                onChange={(e) => setData({ ...data, Type: e.target.value })}
                className="w-full my-2"
                isInvalid={!!errors.Type}
                errorMessage={errors.Type}
              >
                <SelectItem key="Pagbabaybay">
                  Assessment 1: Pagbabaybay
                </SelectItem>
                <SelectItem key="Pantig" isDisabled={!assessmentExist["Assessment 1"]}>
                  Assessment 2: Pantig
                </SelectItem>
                <SelectItem key="Salita" isDisabled={!assessmentExist["Assessment 2"]}>
                  Assessment 3: Salita
                </SelectItem>
                <SelectItem key="Pagbabasa" isDisabled={!assessmentExist["Assessment 3"]}>
                  Assessment 4: Pagbabasa
                </SelectItem>
              </Select>

              {data.Type === "Pagbabasa" && (
                <Select
                  labelPlacement="outside"
                  label="Title"
                  value={data.Title}
                  placeholder="Select a Title:"
                  onChange={(e) => setData({ ...data, Title: e.target.value })}
                  className="w-full my-2"
                  isInvalid={!!errors.Title}
                  errorMessage={errors.Title}
                >
                  {sentences.map((sentence) => (
                    <SelectItem key={sentence.Title} value={sentence.Title}>
                      {sentence.Title}
                    </SelectItem>
                  ))}
                </Select>
              )}

              {data.Type !== "Pagbabasa" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[...Array(10)].map((_, i) => (
                    <Select
                      key={`Item${i + 1}`}
                      labelPlacement="outside"
                      label={`Item ${i + 1}`}
                      value={data[`Item${i + 1}`]}
                      placeholder="Select a word:"
                      onChange={(e) =>
                        handleWordChange(`Item${i + 1}`, e.target.value)
                      }
                      className="w-full my-2"
                      isInvalid={!!errors[`Item${i + 1}`]}
                      errorMessage={errors[`Item${i + 1}`]}
                    >
                      {filteredWords.map((word) => (
                        <SelectItem key={word.ItemCode} value={word.Word}>
                          {word.Word}
                        </SelectItem>
                      ))}
                    </Select>
                  ))}
                </div>
              )}
            </ModalBody>
            <ModalFooter className="flex justify-end space-x-2">
              <Button color="danger" variant="light" onClick={handleClose}>
                Close
              </Button>
              <Button color="primary" type="submit">
                Create Activity
              </Button>
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
      <CreateSuccess show={modalShow} onHide={() => setModalShow(false)} />
    </>
  );
};

CreateAssessment.propTypes = {
  show: PropTypes.bool.isRequired,
  handleClose: PropTypes.func.isRequired,
};

export default CreateAssessment;
