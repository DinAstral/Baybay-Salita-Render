import PropTypes from "prop-types"; // Import PropTypes
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
} from "@nextui-org/react";
import { Link } from "react-router-dom";

const ViewAssessment = ({ show, onHide, activity }) => {
  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="contained-modal-title-vcenter"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col" id="edit-user-modal-title">
          View Assessment Information
        </ModalHeader>
        <ModalBody>
          <p>Do you want to view the assessment of the student selected?</p>
        </ModalBody>
        <ModalFooter>
          <Button color="danger" variant="light" onClick={onHide}>
            Cancel
          </Button>
          <Link to={`/adminViewAssessment/${activity?.ActivityCode}`}>
            <Button color="primary">View</Button>
          </Link>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

ViewAssessment.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  activity: PropTypes.object,
};

export default ViewAssessment;
