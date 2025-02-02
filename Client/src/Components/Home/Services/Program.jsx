import "./Program.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faMobileScreen,
  faBook,
  faLightbulb,
} from "@fortawesome/free-solid-svg-icons";
import service_1 from "../../../assets/img23.png";
import service_2 from "../../../assets/img4.png";
import service_3 from "../../../assets/img5.png";

const Program = () => {
  return (
    <div className="services">
      <div className="service">
        <img src={service_1} alt="" />
        <div className="caption">
          <FontAwesomeIcon
            icon={faMobileScreen}
            size="4x"
            inverse
            className="icon"
          />
          <p>Madaling Pag-gamit ng Aplikasyon</p>
        </div>
      </div>

      <div className="service">
        <img src={service_2} alt="" />
        <div className="caption">
          <FontAwesomeIcon icon={faBook} size="4x" inverse className="icon" />
          <p>Tulong Sa Pagbaybay ng Salita ng mga bata</p>
        </div>
      </div>

      <div className="service">
        <img src={service_3} alt="" />
        <div className="caption">
          <FontAwesomeIcon
            icon={faLightbulb}
            size="4x"
            inverse
            className="icon"
          />
          <p>Pagpapalawak sa Kaalaman ng mga bata</p>
        </div>
      </div>
    </div>
  );
};

export default Program;
